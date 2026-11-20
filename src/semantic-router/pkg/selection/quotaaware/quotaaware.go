// Package quotaaware registers the "quota_aware" decision algorithm: a
// selector that ranks a decision's candidate models by live quota headroom
// reported by one or more CLIProxyAPI instances, picking the richest
// candidate still within its configured minimum headroom, falling back to
// the first candidate with no reported quota ceiling ("metered" models).
package quotaaware

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const algorithmType = "quota_aware"

func init() {
	err := config.RegisterDecisionAlgorithm(config.NewDecisionAlgorithmType(
		config.AlgorithmCatalogEntry{
			Type:        algorithmType,
			DisplayName: "Quota Aware",
			Description: "Rank candidates by live per-model quota headroom reported by CLIProxyAPI instances.",
			Tier:        "experimental",
		},
		config.AlgorithmOptions[Config]{
			Defaults: func(cfg *Config) {
				if cfg.TimeoutSeconds == 0 {
					cfg.TimeoutSeconds = 3
				}
				if cfg.CacheTTLSeconds == 0 {
					cfg.CacheTTLSeconds = 5
				}
				if cfg.MinHeadroomPercent == 0 {
					cfg.MinHeadroomPercent = 10
				}
			},
			Validate: func(decision string, cfg *Config) error {
				if len(cfg.Sources) == 0 {
					return fmt.Errorf("quota_aware requires at least one entry in 'sources'")
				}
				for i, source := range cfg.Sources {
					if source.ModelPrefix == "" {
						return fmt.Errorf("sources[%d]: model_prefix is required", i)
					}
					if source.QuotaSummaryURL == "" {
						return fmt.Errorf("sources[%d]: quota_summary_url is required", i)
					}
				}
				return nil
			},
		},
	))
	if err != nil {
		panic(fmt.Sprintf("quotaaware: failed to register %q decision algorithm: %v", algorithmType, err))
	}
}

// Source maps one CLIProxyAPI instance's quota-summary endpoint to the model
// name prefix it serves (e.g. "cpa-claude/" -> that instance's own Service).
type Source struct {
	ModelPrefix     string `yaml:"model_prefix" json:"model_prefix"`
	QuotaSummaryURL string `yaml:"quota_summary_url" json:"quota_summary_url"`
	// AccessKeyEnv names the environment variable holding the quota-summary
	// endpoint's bearer token, the same convention as a backend_ref's
	// auth.api_key_env: config.yaml never carries the secret itself.
	AccessKeyEnv string `yaml:"access_key_env" json:"access_key_env,omitempty"`
}

// Config is the "quota_aware" algorithm block's schema.
type Config struct {
	Sources            []Source `yaml:"sources" json:"sources"`
	MinHeadroomPercent float64  `yaml:"min_headroom_percent" json:"min_headroom_percent,omitempty"`
	TimeoutSeconds     int      `yaml:"timeout_seconds" json:"timeout_seconds,omitempty"`
	CacheTTLSeconds    int      `yaml:"cache_ttl_seconds" json:"cache_ttl_seconds,omitempty"`

	cacheOnce sync.Once
	cache     *quotaCache
}

type quotaSummaryResponse struct {
	Quotas []quotaEntry `json:"quotas"`
}

type quotaEntry struct {
	Model       string   `json:"model"`
	Available   bool     `json:"available"`
	PercentUsed *float64 `json:"percent_used,omitempty"`
}

// quotaCache memoizes each source's quota-summary response for CacheTTLSeconds,
// so a burst of concurrent requests triggers at most one fetch per source.
type quotaCache struct {
	mu      sync.Mutex
	ttl     time.Duration
	timeout time.Duration
	entries map[string]cacheEntry
}

type cacheEntry struct {
	fetchedAt time.Time
	byModel   map[string]quotaEntry
}

func (c *quotaCache) headroom(ctx context.Context, source Source) map[string]quotaEntry {
	c.mu.Lock()
	entry, ok := c.entries[source.QuotaSummaryURL]
	if ok && time.Since(entry.fetchedAt) < c.ttl {
		c.mu.Unlock()
		return entry.byModel
	}
	c.mu.Unlock()

	byModel := fetchQuotaSummary(ctx, source, c.timeout)

	c.mu.Lock()
	c.entries[source.QuotaSummaryURL] = cacheEntry{fetchedAt: time.Now(), byModel: byModel}
	c.mu.Unlock()
	return byModel
}

func fetchQuotaSummary(ctx context.Context, source Source, timeout time.Duration) map[string]quotaEntry {
	byModel := make(map[string]quotaEntry)
	reqCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, source.QuotaSummaryURL, nil)
	if err != nil {
		return byModel
	}
	if source.AccessKeyEnv != "" {
		if key := os.Getenv(source.AccessKeyEnv); key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return byModel
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return byModel
	}

	var decoded quotaSummaryResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return byModel
	}
	for _, entry := range decoded.Quotas {
		if existing, ok := byModel[entry.Model]; !ok || betterQuota(entry, existing) {
			byModel[entry.Model] = entry
		}
	}
	return byModel
}

// betterQuota prefers an available auth over an exhausted one, then the one
// with more headroom, when a model is reachable through more than one auth
// on the same CLIProxyAPI instance.
func betterQuota(candidate, existing quotaEntry) bool {
	if candidate.Available != existing.Available {
		return candidate.Available
	}
	if candidate.PercentUsed == nil {
		return false
	}
	if existing.PercentUsed == nil {
		return true
	}
	return *candidate.PercentUsed < *existing.PercentUsed
}

func (c *Config) cacheFor() *quotaCache {
	c.cacheOnce.Do(func() {
		c.cache = &quotaCache{
			ttl:     time.Duration(c.CacheTTLSeconds) * time.Second,
			timeout: time.Duration(c.TimeoutSeconds) * time.Second,
			entries: make(map[string]cacheEntry),
		}
	})
	return c.cache
}

func (c *Config) sourceFor(model string) (Source, bool) {
	for _, source := range c.Sources {
		if strings.HasPrefix(model, source.ModelPrefix) {
			return source, true
		}
	}
	return Source{}, false
}

type scoredCandidate struct {
	ref       config.ModelRef
	headroom  float64 // percent remaining, 100 for unmetered/unknown models
	available bool
	metered   bool
}

// Select implements selection.ExtensionAlgorithm: it ranks CandidateModels by
// live quota headroom and returns the best one still available, preferring
// metered candidates above MinHeadroomPercent, then any available candidate
// by headroom, as a decision's quota-exhausted fallback.
func (c *Config) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	cache := c.cacheFor()
	scored := make([]scoredCandidate, 0, len(selCtx.CandidateModels))

	for _, ref := range selCtx.CandidateModels {
		source, ok := c.sourceFor(ref.Model)
		if !ok {
			scored = append(scored, scoredCandidate{ref: ref, headroom: 100, available: true, metered: false})
			continue
		}
		byModel := cache.headroom(ctx, source)
		entry, found := byModel[ref.Model]
		if !found {
			scored = append(scored, scoredCandidate{ref: ref, headroom: 100, available: true, metered: false})
			continue
		}
		headroom := 100.0
		if entry.PercentUsed != nil {
			headroom = 100 - *entry.PercentUsed
		}
		scored = append(scored, scoredCandidate{ref: ref, headroom: headroom, available: entry.Available, metered: true})
	}

	chosen, reasoning := pickBest(scored, c.MinHeadroomPercent)
	if chosen == nil {
		return nil, fmt.Errorf("quota_aware: no available candidate among %d", len(selCtx.CandidateModels))
	}

	return &selection.SelectionResult{
		SelectedModel:     chosen.ref.Model,
		SelectedCandidate: &chosen.ref,
		LoRAName:          chosen.ref.LoRAName,
		Score:             chosen.headroom,
		Confidence:        1,
		Reasoning:         reasoning,
	}, nil
}

// pickBest prefers an unmetered candidate (order given) over a metered one,
// since an unmetered model has no quota ceiling to protect; among metered
// candidates it picks the most headroom above minHeadroom, falling back to
// the single candidate with the most headroom when every metered candidate
// is below minHeadroom.
func pickBest(scored []scoredCandidate, minHeadroom float64) (*scoredCandidate, string) {
	for i := range scored {
		if !scored[i].metered && scored[i].available {
			return &scored[i], fmt.Sprintf("%s has no quota ceiling", scored[i].ref.Model)
		}
	}

	available := make([]scoredCandidate, 0, len(scored))
	for _, candidate := range scored {
		if candidate.available {
			available = append(available, candidate)
		}
	}
	if len(available) == 0 {
		return nil, ""
	}
	sort.Slice(available, func(i, j int) bool { return available[i].headroom > available[j].headroom })

	for i := range available {
		if available[i].headroom >= minHeadroom {
			return &available[i], fmt.Sprintf("%s has %.1f%% quota headroom", available[i].ref.Model, available[i].headroom)
		}
	}
	best := available[0]
	return &best, fmt.Sprintf("%s has the most headroom (%.1f%%) though below the %.1f%% minimum", best.ref.Model, best.headroom, minHeadroom)
}
