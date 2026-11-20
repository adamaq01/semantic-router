package quotaaware

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func candidate(model string) config.ModelRef {
	return config.ModelRef{Model: model}
}

func percent(value float64) *float64 { return &value }

func TestPickBestPrefersUnmetered(t *testing.T) {
	scored := []scoredCandidate{
		{ref: candidate("metered"), headroom: 5, available: true},
		{ref: candidate("free"), headroom: 100, available: true, unmetered: true},
	}
	chosen, _ := pickBest(scored, 10)
	if chosen == nil || chosen.ref.Model != "free" {
		t.Fatalf("expected unmetered candidate to win, got %+v", chosen)
	}
}

func TestPickBestRanksByHeadroomAboveMinimum(t *testing.T) {
	scored := []scoredCandidate{
		{ref: candidate("low"), headroom: 5, available: true},
		{ref: candidate("high"), headroom: 80, available: true},
		{ref: candidate("mid"), headroom: 40, available: true},
	}
	chosen, _ := pickBest(scored, 10)
	if chosen == nil || chosen.ref.Model != "high" {
		t.Fatalf("expected the highest-headroom candidate, got %+v", chosen)
	}
}

func TestPickBestFallsBackWhenAllBelowMinimum(t *testing.T) {
	scored := []scoredCandidate{
		{ref: candidate("a"), headroom: 2, available: true},
		{ref: candidate("b"), headroom: 5, available: true},
	}
	chosen, _ := pickBest(scored, 10)
	if chosen == nil || chosen.ref.Model != "b" {
		t.Fatalf("expected the least-exhausted candidate as fallback, got %+v", chosen)
	}
}

func TestPickBestSkipsUnavailableCandidates(t *testing.T) {
	scored := []scoredCandidate{
		{ref: candidate("down"), headroom: 100, available: false},
		{ref: candidate("up"), headroom: 20, available: true},
	}
	chosen, _ := pickBest(scored, 10)
	if chosen == nil || chosen.ref.Model != "up" {
		t.Fatalf("expected the only available candidate, got %+v", chosen)
	}
}

func TestPickBestReturnsNilWhenNoneAvailable(t *testing.T) {
	scored := []scoredCandidate{
		{ref: candidate("a"), headroom: 100, available: false},
	}
	chosen, _ := pickBest(scored, 10)
	if chosen != nil {
		t.Fatalf("expected no candidate, got %+v", chosen)
	}
}

func TestPickBestReturnsNilOnEmptyInput(t *testing.T) {
	chosen, _ := pickBest(nil, 10)
	if chosen != nil {
		t.Fatalf("expected no candidate, got %+v", chosen)
	}
}

func TestPickBestTiesBreakByCandidateOrder(t *testing.T) {
	scored := []scoredCandidate{
		{ref: candidate("first"), headroom: 50, available: true},
		{ref: candidate("second"), headroom: 50, available: true},
	}
	chosen, _ := pickBest(scored, 10)
	if chosen == nil || chosen.ref.Model != "first" {
		t.Fatalf("expected the first of equal candidates, got %+v", chosen)
	}
}

func TestBetterQuotaPrefersAvailable(t *testing.T) {
	exhausted := quotaEntry{Available: false, PercentUsed: percent(10)}
	available := quotaEntry{Available: true, PercentUsed: percent(90)}
	if !betterQuota(available, exhausted) {
		t.Fatal("expected an available entry to beat an exhausted one regardless of percent")
	}
	if betterQuota(exhausted, available) {
		t.Fatal("expected an exhausted entry not to beat an available one")
	}
}

func TestBetterQuotaPrefersMoreHeadroom(t *testing.T) {
	low := quotaEntry{Available: true, PercentUsed: percent(80)}
	high := quotaEntry{Available: true, PercentUsed: percent(20)}
	if !betterQuota(high, low) {
		t.Fatal("expected the entry with more headroom to win")
	}
	if betterQuota(low, high) {
		t.Fatal("expected the entry with less headroom to lose")
	}
}

func TestBetterQuotaPrefersKnownPercentOverNil(t *testing.T) {
	known := quotaEntry{Available: true, PercentUsed: percent(50)}
	unknown := quotaEntry{Available: true, PercentUsed: nil}
	if !betterQuota(known, unknown) {
		t.Fatal("expected a known percent to beat an unknown one")
	}
	if betterQuota(unknown, known) {
		t.Fatal("expected an unknown percent not to beat a known one")
	}
}
