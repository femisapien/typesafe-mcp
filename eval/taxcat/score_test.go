package main

import (
	"math"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestScoreRun(t *testing.T) {
	rows := []payee{
		{Row: 1, Payee: "Kaiser", Label: "medical", Origin: "real"},
		{Row: 2, Payee: "Red Cross", Label: "charitable", Origin: "added", Hard: true},
		{Row: 3, Payee: "Starbucks", Label: "personal", Origin: "real"},
		{Row: 4, Payee: "IRS", Label: "federal_tax", Origin: "added"},
	}
	ans := []answer{
		{Row: 1, Choice: "medical", Confidence: 0.9, Probabilities: map[string]float64{"medical": 0.95}},
		{Row: 2, Choice: "personal", Confidence: 0.4, Probabilities: map[string]float64{"personal": 0.6}},
		{Row: 3, Choice: "personal", Confidence: 0.8, Probabilities: map[string]float64{"personal": 0.85}},
		// row 4 failed upstream and has no answer
	}
	r := scoreRun("t", rows, ans)
	near := func(name string, got, want float64) {
		t.Helper()
		if math.Abs(got-want) > 1e-9 {
			t.Errorf("%s = %v, want %v", name, got, want)
		}
	}
	near("Acc", r.Acc, 0.5)
	near("AccReal", r.AccReal, 1)
	near("AccAdd", r.AccAdd, 0)
	near("AccHard", r.AccHard, 0)
	near("DeductibleRecall", r.DeductibleRecall, 0.5) // medical hit, charitable missed
	near("Precision[personal]", r.Precision["personal"], 0.5)
	// F1: medical 1, charitable 0, personal 2/3, federal_tax 0.
	near("MacroF1", r.MacroF1, (1+0+2.0/3+0)/4)
	if len(r.Missed) != 2 || r.Missed[0].Payee != "Red Cross" {
		t.Errorf("Missed = %+v, want Red Cross first, then IRS", r.Missed)
	}
	// ≥0.5 keeps rows 1 and 3, both right; row 2 is under, row 4 has no answer.
	near("coverage@0.5", r.Coverage[1].Coverage, 0.5)
	near("acc@0.5", r.Coverage[1].Acc, 1)
}

func TestArgmax(t *testing.T) {
	p, best, conf := argmax(map[string]apiAnswer{"medical": {Noul: 0.6}, "personal": {Noul: 0.2}})
	if best != "medical" || math.Abs(p["medical"]-0.75) > 1e-9 || math.Abs(conf-0.5) > 1e-9 {
		t.Errorf("argmax = %v %q %v, want medical at 0.75, confidence 0.5", p, best, conf)
	}
}

func TestApprox(t *testing.T) {
	for in, want := range map[float64]string{-2: "under $5 charge", -20: "$20 charge", 2800: "$2,800 deposit"} {
		if got := approx(in); got != want {
			t.Errorf("approx(%v) = %q, want %q", in, got, want)
		}
	}
}

func TestLabelsInCriteriaOrder(t *testing.T) {
	if len(labels) != 11 || labels[0] != "wage_income" || labels[10] != "none_of_these" {
		t.Errorf("labels = %v", labels)
	}
}

func TestAverage(t *testing.T) {
	a := average("jev", []report{
		{Acc: 0.9, Abstained: 2, Recall: map[string]float64{"medical": 1}, Coverage: []coverage{{0.5, 0.8, 1}},
			Confusions: []confusion{{"medical", "personal", 2}}, Missed: []missed{{"CVS", "medical", "personal", 0.4}}},
		{Acc: 0.8, Abstained: 4, Recall: map[string]float64{"medical": 0.5}, Coverage: []coverage{{0.5, 0.6, 0.5}},
			Missed: []missed{{"CVS", "medical", "personal", 0.7}}},
	})
	if a.Passes != 2 || math.Abs(a.Acc-0.85) > 1e-9 || a.AccMin != 0.8 || a.AccMax != 0.9 || a.Abstained != 3 {
		t.Errorf("average = passes %d acc %v [%v, %v] abstained %v", a.Passes, a.Acc, a.AccMin, a.AccMax, a.Abstained)
	}
	if math.Abs(a.Recall["medical"]-0.75) > 1e-9 || math.Abs(a.Coverage[0].Coverage-0.7) > 1e-9 || a.Confusions[0].N != 1 {
		t.Errorf("average recall %v coverage %+v confusions %+v", a.Recall, a.Coverage, a.Confusions)
	}
	if len(a.Missed) != 1 || a.Missed[0].Conf != 0.7 {
		t.Errorf("Missed = %+v, want CVS once at 0.7", a.Missed)
	}
}

func TestCostPer100(t *testing.T) {
	p := &price{Prompt: 0.04e-6, Completion: 1e-6}
	calls := []callMeta{
		{InputTokens: 1_000_000, OutputTokens: 10_000, ItemCount: 500},
		{InputTokens: 1_000_000, OutputTokens: 10_000, ItemCount: 500},
	}
	// 2M × $0.04/M + 20k × $1/M = $0.10 for 1,000 items, so $0.01 per 100.
	if got, ok := costPer100([]runMeta{{Price: p, Calls: calls}}); !ok || math.Abs(got-0.01) > 1e-12 {
		t.Errorf("costPer100 = %v, %v; want 0.01, true", got, ok)
	}
	if _, ok := costPer100([]runMeta{{Calls: calls}}); ok {
		t.Error("a run with no saved price got a cost")
	}
	if _, ok := costPer100([]runMeta{{Price: p}}); ok {
		t.Error("a run with no calls got a cost")
	}
	// Each pass is priced at its own rate: pass 2 at double the rate makes
	// $0.10 + $0.20 for 2,000 items, so $0.015 per 100. A pass with no saved
	// rate is left out rather than blanking the group.
	p2 := &price{Prompt: 0.08e-6, Completion: 2e-6}
	passes := []runMeta{{Calls: calls}, {Price: p, Calls: calls}, {Price: p2, Calls: calls}}
	if got, ok := costPer100(passes); !ok || math.Abs(got-0.015) > 1e-12 {
		t.Errorf("costPer100 over passes = %v, %v; want 0.015, true", got, ok)
	}
}

func TestMatchPrice(t *testing.T) {
	models := []orModel{
		{ID: "liquid/d1", CanonicalSlug: "liquid/d1-20260930"},
		{ID: "typesafe/jev-router", CanonicalSlug: "typesafe/jev-router"},
	}
	models[0].Pricing.Prompt, models[0].Pricing.Completion = "0.00000004", "0"
	models[1].Pricing.Prompt, models[1].Pricing.Completion = "-1", "-1"
	for _, model := range []string{"liquid/d1", "liquid/d1-20260930"} {
		if p, err := matchPrice(models, model); err != nil || p.ID != "liquid/d1" || p.Prompt != 4e-8 {
			t.Errorf("matchPrice(%q) = %+v, %v; want liquid/d1 at 4e-8", model, p, err)
		}
	}
	for _, model := range []string{"jev-1.13.0", "typesafe/jev-router"} {
		if p, err := matchPrice(models, model); err == nil {
			t.Errorf("matchPrice(%q) = %+v, want an error", model, p)
		}
	}
}

func TestFetchPrice(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Write([]byte(`{"data":[{"id":"cloudflare/clef","canonical_slug":"cloudflare/clef","pricing":{"prompt":"0.00000024","completion":"0"}}]}`))
	}))
	defer srv.Close()
	defer func(u string) { openRouterModels = u }(openRouterModels)
	openRouterModels = srv.URL
	p, err := fetchPrice("cloudflare/clef")
	if err != nil || p.Prompt != 2.4e-7 || p.FetchedAt == "" {
		t.Errorf("fetchPrice = %+v, %v", p, err)
	}
}
