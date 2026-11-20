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
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

const algorithmType = "quota_aware"

// failureBackoff bounds how often a failing quota-summary endpoint is retried,
// independent of CacheTTLSeconds: a short, fixed backoff rather than caching
// the failure for the full (user-configurable, possibly much longer) TTL.
const failureBackoff = time.Second

func init() {
	err := config.RegisterDecisionAlgorithm(config.NewDecisionAlgorithmType(
		config.AlgorithmCatalogEntry{
			Type:        algorithmType,
			DisplayName: "Quota Aware",
			Description: "Rank candidates by live per-model quota headroom reported by CLIProxyAPI instances.",
			Tier:        "experimental",
		},
		config.AlgorithmOptions[Config]{
			Strict: true,
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
					if source.QuotaSummaryURL == "" {
						return fmt.Errorf("sources[%d]: quota_summary_url is required", i)
					}
					if len(source.Models) == 0 {
						return fmt.Errorf("sources[%d]: models requires at least one entry", i)
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

// Source is one CLIProxyAPI instance's quota-summary endpoint. Models maps
// this decision's candidate name (ModelRef.Model, e.g. "cpa-claude-opus") to
// the bare model id CLIProxyAPI's own quota-summary reports it under (e.g.
// "claude-opus-5-5", with no provider/instance prefix): the two are unrelated
// strings with no derivable relationship, so the mapping must be explicit.
type Source struct {
	QuotaSummaryURL string            `yaml:"quota_summary_url" json:"quota_summary_url"`
	Models          map[string]string `yaml:"models" json:"models"`
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

// quotaCache memoizes each source's quota-summary response for CacheTTLSeconds.
// Each source gets its own mutex, held across its fetch: concurrent requests
// for the same source block on one fetch rather than each starting their own.
type quotaCache struct {
	mu      sync.Mutex
	ttl     time.Duration
	timeout time.Duration
	entries map[string]*sourceCache
}

type sourceCache struct {
	mu                sync.Mutex
	lastGood          *cacheEntry
	lastAttempt       time.Time
	lastAttemptFailed bool
}

type cacheEntry struct {
	fetchedAt time.Time
	byModel   map[string]quotaEntry
}

// headroom returns a source's cached quota-summary, refetching when stale. A
// failed fetch never overwrites a previous good result: it logs and falls
// back to the last known-good data (even if past its TTL) rather than
// treating every candidate as freshly unmetered for the rest of the TTL.
func (c *quotaCache) headroom(ctx context.Context, source Source) map[string]quotaEntry {
	c.mu.Lock()
	sc, ok := c.entries[source.QuotaSummaryURL]
	if !ok {
		sc = &sourceCache{}
		c.entries[source.QuotaSummaryURL] = sc
	}
	c.mu.Unlock()

	sc.mu.Lock()
	defer sc.mu.Unlock()

	if sc.lastGood != nil && time.Since(sc.lastGood.fetchedAt) < c.ttl {
		return sc.lastGood.byModel
	}
	if sc.lastAttemptFailed && time.Since(sc.lastAttempt) < failureBackoff {
		return lastGoodByModel(sc)
	}

	// Detached from the triggering request's context: one client disconnecting
	// must not fail a fetch that other concurrent requests are waiting on.
	fetchCtx, cancel := context.WithTimeout(context.WithoutCancel(ctx), c.timeout)
	defer cancel()
	byModel, err := fetchQuotaSummary(fetchCtx, source)
	sc.lastAttempt = time.Now()
	if err != nil {
		sc.lastAttemptFailed = true
		logging.Warnf("quotaaware: quota-summary fetch failed for %s: %v", source.QuotaSummaryURL, err)
		return lastGoodByModel(sc)
	}
	sc.lastAttemptFailed = false
	sc.lastGood = &cacheEntry{fetchedAt: time.Now(), byModel: byModel}
	return byModel
}

func lastGoodByModel(sc *sourceCache) map[string]quotaEntry {
	if sc.lastGood == nil {
		return nil
	}
	return sc.lastGood.byModel
}

func fetchQuotaSummary(ctx context.Context, source Source) (map[string]quotaEntry, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, source.QuotaSummaryURL, nil)
	if err != nil {
		return nil, fmt.Errorf("build request: %w", err)
	}
	if source.AccessKeyEnv != "" {
		if key := os.Getenv(source.AccessKeyEnv); key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
	}
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("request failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status %d", resp.StatusCode)
	}

	var decoded quotaSummaryResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}
	byModel := make(map[string]quotaEntry, len(decoded.Quotas))
	for _, entry := range decoded.Quotas {
		if existing, ok := byModel[entry.Model]; !ok || betterQuota(entry, existing) {
			byModel[entry.Model] = entry
		}
	}
	return byModel, nil
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
			entries: make(map[string]*sourceCache),
		}
	})
	return c.cache
}

// sourceFor finds the source whose Models map knows candidateName, returning
// the source and the bare model id to look its quota up under.
func (c *Config) sourceFor(candidateName string) (Source, string, bool) {
	for _, source := range c.Sources {
		if quotaModel, ok := source.Models[candidateName]; ok {
			return source, quotaModel, true
		}
	}
	return Source{}, "", false
}

type scoredCandidate struct {
	ref       config.ModelRef
	headroom  float64 // percent remaining
	available bool
	// unmetered is true only for a candidate no source's Models map names at
	// all: it has no quota ceiling to protect, by design, and always wins.
	// A candidate CLIProxyAPI simply hasn't reported usage for yet (it builds
	// quota-summary entries lazily, per auth, on first use) is metered with
	// unknown (optimistic) headroom instead: it must not automatically beat
	// every candidate that already has real usage data.
	unmetered bool
}

// Select implements selection.ExtensionAlgorithm: it ranks CandidateModels by
// live quota headroom and returns the best one still available, preferring
// metered candidates above MinHeadroomPercent, then any available candidate
// by headroom, as a decision's quota-exhausted fallback.
func (c *Config) Select(ctx context.Context, selCtx *selection.SelectionContext) (*selection.SelectionResult, error) {
	cache := c.cacheFor()
	scored := make([]scoredCandidate, 0, len(selCtx.CandidateModels))

	for _, ref := range selCtx.CandidateModels {
		source, quotaModel, ok := c.sourceFor(ref.Model)
		if !ok {
			scored = append(scored, scoredCandidate{ref: ref, headroom: 100, available: true, unmetered: true})
			continue
		}
		byModel := cache.headroom(ctx, source)
		entry, found := byModel[quotaModel]
		if !found {
			scored = append(scored, scoredCandidate{ref: ref, headroom: 100, available: true})
			continue
		}
		headroom := 100.0
		if entry.PercentUsed != nil {
			headroom = 100 - *entry.PercentUsed
		}
		scored = append(scored, scoredCandidate{ref: ref, headroom: headroom, available: entry.Available})
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

// pickBest prefers a candidate no source maps at all (order given), since it
// has no quota ceiling to protect; among the rest (including one CLIProxyAPI
// simply hasn't reported usage for yet) it picks the most headroom above
// minHeadroom, falling back to the single candidate with the most headroom
// when every one of them is below minHeadroom.
func pickBest(scored []scoredCandidate, minHeadroom float64) (*scoredCandidate, string) {
	for i := range scored {
		if scored[i].unmetered && scored[i].available {
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
	// Stable: candidate order (as declared in modelRefs) is the explicit
	// tiebreak between equal headroom, e.g. multiple models on one
	// subscription reporting the same unified rate-limit utilization.
	sort.SliceStable(available, func(i, j int) bool { return available[i].headroom > available[j].headroom })

	for i := range available {
		if available[i].headroom >= minHeadroom {
			return &available[i], fmt.Sprintf("%s has %.1f%% quota headroom", available[i].ref.Model, available[i].headroom)
		}
	}
	best := available[0]
	return &best, fmt.Sprintf("%s has the most headroom (%.1f%%) though below the %.1f%% minimum", best.ref.Model, best.headroom, minHeadroom)
}
