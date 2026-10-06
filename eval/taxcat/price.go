package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// openRouterModels lists OpenRouter's decision models with their prices. It
// needs no key. A var so tests can point it at a stub.
var openRouterModels = "https://openrouter.ai/api/v1/models?output_modalities=decisions"

// price is the OpenRouter rate a run was billed at, saved in its meta.json so
// scoring never needs the network and old runs keep their old rates.
type price struct {
	ID         string  `json:"id"`
	Prompt     float64 `json:"prompt"`     // USD per input token
	Completion float64 `json:"completion"` // USD per output token
	FetchedAt  string  `json:"fetched_at"`
}

type orModel struct {
	ID            string `json:"id"`
	CanonicalSlug string `json:"canonical_slug"`
	Pricing       struct {
		Prompt     string `json:"prompt"`
		Completion string `json:"completion"`
	} `json:"pricing"`
}

// fetchPrice looks up the model a reply named in OpenRouter's catalog.
func fetchPrice(model string) (*price, error) {
	c := http.Client{Timeout: 30 * time.Second}
	resp, err := c.Get(openRouterModels)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("%s: %s", openRouterModels, resp.Status)
	}
	var body struct {
		Data []orModel `json:"data"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&body); err != nil {
		return nil, fmt.Errorf("decoding OpenRouter models: %w", err)
	}
	p, err := matchPrice(body.Data, model)
	if err != nil {
		return nil, err
	}
	p.FetchedAt = time.Now().UTC().Format(time.RFC3339)
	return p, nil
}

// matchPrice finds model by id, else by canonical slug, which is what replies
// name (liquid/d1-20260930 for liquid/d1). A slug shared with a :free variant
// resolves to whichever comes first; no model in this eval has one.
func matchPrice(models []orModel, model string) (*price, error) {
	var hit *orModel
	for i, m := range models {
		if m.ID == model {
			hit = &models[i]
			break
		}
		if hit == nil && m.CanonicalSlug == model {
			hit = &models[i]
		}
	}
	if hit == nil {
		return nil, fmt.Errorf("model %q is not in OpenRouter's decision models", model)
	}
	in, err1 := strconv.ParseFloat(hit.Pricing.Prompt, 64)
	out, err2 := strconv.ParseFloat(hit.Pricing.Completion, 64)
	// OpenRouter marks routers, whose price depends on the model they pick, as -1.
	if err1 != nil || err2 != nil || in < 0 || out < 0 {
		return nil, fmt.Errorf("model %q has no fixed price (%q in, %q out)", model, hit.Pricing.Prompt, hit.Pricing.Completion)
	}
	return &price{ID: hit.ID, Prompt: in, Completion: out}, nil
}

// costPer100 prices each pass's calls at the rate that pass saved and scales
// the total to 100 categorized items. Passes with no saved rate are left out;
// it reports false when no pass has a rate or the priced passes have no items.
func costPer100(passes []runMeta) (float64, bool) {
	var usd float64
	var items int
	for _, m := range passes {
		if m.Price == nil {
			continue
		}
		for _, c := range m.Calls {
			usd += float64(c.InputTokens)*m.Price.Prompt + float64(c.OutputTokens)*m.Price.Completion
			items += c.ItemCount
		}
	}
	if items == 0 {
		return 0, false
	}
	return usd * 100 / float64(items), true
}
