package main

import (
	"cmp"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
)

// deductible are the classes where a miss costs the taxpayer money.
var deductible = []string{"medical", "charitable", "state_local_tax", "education", "student_loan_interest"}

var thresholds = []float64{0.3, 0.5, 0.7}

// report holds one run's scores, or the mean over several passes of a run.
// Passes, AccMin and AccMax describe the spread; counts are per-pass means.
type report struct {
	Name                  string
	N, Passes             int
	Errors, Abstained     float64
	Acc, AccReal, AccAdd  float64
	AccHard, MacroF1, ECE float64
	AccMin, AccMax        float64
	DeductibleRecall      float64
	Precision, Recall     map[string]float64
	Support               map[string]int
	Confusions            []confusion
	Missed                []missed
	Coverage              []coverage
	Meta                  runMeta
}

type confusion struct {
	Want, Got string
	N         float64
}

type missed struct {
	Payee, Want, Got string
	Conf             float64
}

type coverage struct {
	Threshold, Coverage, Acc float64
}

type runMeta struct {
	Profile    string `json:"profile"`
	Protocol   string `json:"protocol"`
	WithAmount bool   `json:"with_amount"`
	Calls      []struct {
		Model        string `json:"model"`
		InputTokens  int    `json:"input_tokens"`
		OutputTokens int    `json:"output_tokens"`
		ItemCount    int    `json:"item_count"`
		LatencyMS    int64  `json:"latency_ms"`
	} `json:"calls"`
	Errors map[string]string `json:"errors"`
}

func score(args []string) error {
	fs := flag.NewFlagSet("score", flag.ExitOnError)
	data := fs.String("data", "data", "dataset directory")
	dir := fs.String("results", "results", "results directory")
	limit := fs.Int("limit", 0, "score only the first N rows, matching a run's -limit")
	fs.Parse(args)
	rows, err := readJSONL[payee](filepath.Join(*data, "payees.jsonl"))
	if err != nil {
		return err
	}
	if *limit > 0 && *limit < len(rows) {
		rows = rows[:*limit]
	}
	files, _ := filepath.Glob(filepath.Join(*dir, "*.jsonl"))
	var order []string
	groups := map[string][]report{}
	for _, f := range files {
		ans, err := readJSONL[answer](f)
		if err != nil {
			return err
		}
		name := strings.TrimSuffix(filepath.Base(f), ".jsonl")
		r := scoreRun(name, rows, ans)
		if b, err := os.ReadFile(strings.TrimSuffix(f, ".jsonl") + ".meta.json"); err == nil {
			json.Unmarshal(b, &r.Meta)
		}
		r.Errors = float64(len(r.Meta.Errors))
		base := passSuffix.ReplaceAllString(name, "")
		if groups[base] == nil {
			order = append(order, base)
		}
		groups[base] = append(groups[base], r)
	}
	var reps []report
	for _, base := range order {
		reps = append(reps, average(base, groups[base]))
	}
	writeMarkdown(os.Stdout, reps)
	return nil
}

// passSuffix marks repeat passes of one run: jev.p1, jev.p2 and jev.p3 score as jev.
var passSuffix = regexp.MustCompile(`\.p\d+$`)

// average folds repeat passes of one run into a single report: rates, recalls,
// coverage and counts become per-pass means, confusions too, and the misses
// list keeps each payee/prediction pair once at its highest confidence.
func average(name string, ps []report) report {
	n := float64(len(ps))
	a := report{Name: name, N: ps[0].N, Passes: len(ps), Precision: map[string]float64{}, Recall: map[string]float64{}, Support: ps[0].Support, Meta: ps[0].Meta, AccMin: ps[0].Acc, AccMax: ps[0].Acc}
	a.Meta.Calls = nil
	conf := map[[2]string]float64{}
	seen := map[[2]string]int{}
	for i, p := range ps {
		a.Errors += p.Errors / n
		a.Abstained += p.Abstained / n
		a.Acc += p.Acc / n
		a.AccReal += p.AccReal / n
		a.AccAdd += p.AccAdd / n
		a.AccHard += p.AccHard / n
		a.MacroF1 += p.MacroF1 / n
		a.ECE += p.ECE / n
		a.DeductibleRecall += p.DeductibleRecall / n
		a.AccMin, a.AccMax = min(a.AccMin, p.Acc), max(a.AccMax, p.Acc)
		for l, v := range p.Precision {
			a.Precision[l] += v / n
		}
		for l, v := range p.Recall {
			a.Recall[l] += v / n
		}
		for j, c := range p.Coverage {
			if i == 0 {
				a.Coverage = append(a.Coverage, coverage{Threshold: c.Threshold})
			}
			a.Coverage[j].Coverage += c.Coverage / n
			a.Coverage[j].Acc += c.Acc / n
		}
		for _, c := range p.Confusions {
			conf[[2]string{c.Want, c.Got}] += c.N / n
		}
		for _, m := range p.Missed {
			k := [2]string{m.Payee, m.Got}
			if j, ok := seen[k]; ok {
				a.Missed[j].Conf = max(a.Missed[j].Conf, m.Conf)
				continue
			}
			seen[k] = len(a.Missed)
			a.Missed = append(a.Missed, m)
		}
		a.Meta.Calls = append(a.Meta.Calls, p.Meta.Calls...)
	}
	for k, v := range conf {
		a.Confusions = append(a.Confusions, confusion{k[0], k[1], v})
	}
	slices.SortFunc(a.Confusions, func(x, y confusion) int {
		return cmp.Or(cmp.Compare(y.N, x.N), cmp.Compare(x.Want+x.Got, y.Want+y.Got))
	})
	slices.SortFunc(a.Missed, func(x, y missed) int { return cmp.Compare(y.Conf, x.Conf) })
	return a
}

// scoreRun scores one run. Rows with no answer (upstream errors) count as
// wrong, so a flaky host cannot raise its accuracy by failing hard items.
func scoreRun(name string, rows []payee, ans []answer) report {
	got := map[int]answer{}
	for _, a := range ans {
		got[a.Row] = a
	}
	r := report{Name: name, N: len(rows), Precision: map[string]float64{}, Recall: map[string]float64{}, Support: map[string]int{}}
	tp, fp := map[string]int{}, map[string]int{}
	conf := map[[2]string]float64{}
	var right, real, realRight, add, addRight, hard, hardRight int
	type pt struct {
		p  float64
		ok bool
	}
	var pts []pt
	for _, row := range rows {
		a, ok := got[row.Row]
		pred := a.Choice
		if !ok {
			pred = "(error)"
		}
		hit := pred == row.Label
		r.Support[row.Label]++
		if pred == "none_of_these" {
			r.Abstained++
		}
		if hit {
			right++
			tp[row.Label]++
		} else {
			fp[pred]++
			conf[[2]string{row.Label, pred}]++
			r.Missed = append(r.Missed, missed{row.Payee, row.Label, pred, a.Confidence})
		}
		if row.Origin == "real" {
			real++
			if hit {
				realRight++
			}
		} else {
			add++
			if hit {
				addRight++
			}
		}
		if row.Hard {
			hard++
			if hit {
				hardRight++
			}
		}
		if ok {
			pts = append(pts, pt{a.Probabilities[pred], hit})
		}
	}
	r.Acc = ratio(right, len(rows))
	r.AccReal, r.AccAdd, r.AccHard = ratio(realRight, real), ratio(addRight, add), ratio(hardRight, hard)

	var f1s []float64
	var dedHit, dedAll int
	for _, l := range labels {
		if r.Support[l] == 0 {
			continue
		}
		p, rc := ratio(tp[l], tp[l]+fp[l]), ratio(tp[l], r.Support[l])
		r.Precision[l], r.Recall[l] = p, rc
		f1 := 0.0
		if p+rc > 0 {
			f1 = 2 * p * rc / (p + rc)
		}
		f1s = append(f1s, f1)
		if slices.Contains(deductible, l) {
			dedHit += tp[l]
			dedAll += r.Support[l]
		}
	}
	r.MacroF1 = mean(f1s)
	r.DeductibleRecall = ratio(dedHit, dedAll)

	for k, n := range conf {
		r.Confusions = append(r.Confusions, confusion{k[0], k[1], n})
	}
	slices.SortFunc(r.Confusions, func(a, b confusion) int {
		return cmp.Or(cmp.Compare(b.N, a.N), cmp.Compare(a.Want+a.Got, b.Want+b.Got))
	})
	slices.SortFunc(r.Missed, func(a, b missed) int { return cmp.Compare(b.Conf, a.Conf) })

	// ECE over ten equal-width bins of the chosen option's probability.
	var bins [10]struct {
		n, ok int
		p     float64
	}
	for _, x := range pts {
		i := min(int(x.p*10), 9)
		bins[i].n++
		bins[i].p += x.p
		if x.ok {
			bins[i].ok++
		}
	}
	for _, b := range bins {
		if b.n > 0 {
			r.ECE += float64(b.n) / float64(len(pts)) * math.Abs(b.p/float64(b.n)-float64(b.ok)/float64(b.n))
		}
	}

	for _, t := range thresholds {
		var kept, keptRight int
		for _, row := range rows {
			a, ok := got[row.Row]
			if !ok || a.Confidence < t || a.Choice == "none_of_these" {
				continue
			}
			kept++
			if a.Choice == row.Label {
				keptRight++
			}
		}
		r.Coverage = append(r.Coverage, coverage{t, ratio(kept, len(rows)), ratio(keptRight, kept)})
	}
	return r
}

func ratio(a, b int) float64 {
	if b == 0 {
		return 0
	}
	return float64(a) / float64(b)
}

func mean(xs []float64) float64 {
	var s float64
	for _, x := range xs {
		s += x
	}
	return s / float64(max(len(xs), 1))
}

func pct(x float64) string { return fmt.Sprintf("%.1f%%", 100*x) }

func writeMarkdown(w io.Writer, reps []report) {
	fmt.Fprintln(w, "## Headline")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "Each figure is the mean over the run's passes; counts are per pass.")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Run | Model | Passes | Accuracy | Range | Macro-F1 | Deductible recall | Real payees | Added payees | Hard rows | ECE | Abstained | Errors |")
	fmt.Fprintln(w, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
	for _, r := range reps {
		model := ""
		if len(r.Meta.Calls) > 0 {
			model = r.Meta.Calls[0].Model
		}
		fmt.Fprintf(w, "| `%s` | `%s` | %d | %s | %s–%s | %.3f | %s | %s | %s | %s | %.3f | %.1f | %.1f |\n",
			r.Name, model, r.Passes, pct(r.Acc), pct(r.AccMin), pct(r.AccMax), r.MacroF1, pct(r.DeductibleRecall), pct(r.AccReal), pct(r.AccAdd), pct(r.AccHard), r.ECE, r.Abstained, r.Errors)
	}

	fmt.Fprintln(w, "\n## Recall by category")
	fmt.Fprintln(w)
	fmt.Fprint(w, "| Category | Rows |")
	for _, r := range reps {
		fmt.Fprintf(w, " `%s` |", r.Name)
	}
	fmt.Fprint(w, "\n|---|---|")
	for range reps {
		fmt.Fprint(w, "---|")
	}
	fmt.Fprintln(w)
	if len(reps) > 0 {
		for _, l := range labels {
			if reps[0].Support[l] == 0 {
				continue
			}
			fmt.Fprintf(w, "| `%s` | %d |", l, reps[0].Support[l])
			for _, r := range reps {
				fmt.Fprintf(w, " %s |", pct(r.Recall[l]))
			}
			fmt.Fprintln(w)
		}
	}

	fmt.Fprintln(w, "\n## Coverage when abstaining below a confidence threshold")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Run | ≥0.3 coverage / accuracy | ≥0.5 | ≥0.7 |")
	fmt.Fprintln(w, "|---|---|---|---|")
	for _, r := range reps {
		fmt.Fprintf(w, "| `%s` |", r.Name)
		for _, c := range r.Coverage {
			fmt.Fprintf(w, " %s / %s |", pct(c.Coverage), pct(c.Acc))
		}
		fmt.Fprintln(w)
	}

	fmt.Fprintln(w, "\n## Cost and speed")
	fmt.Fprintln(w)
	fmt.Fprintln(w, "| Run | Input tokens / pass | Output tokens / pass | Wall clock / pass |")
	fmt.Fprintln(w, "|---|---|---|---|")
	for _, r := range reps {
		var in, out int
		var ms int64
		for _, c := range r.Meta.Calls {
			in, out, ms = in+c.InputTokens, out+c.OutputTokens, ms+c.LatencyMS
		}
		p := max(r.Passes, 1)
		fmt.Fprintf(w, "| `%s` | %d | %d | %.0fs |\n", r.Name, in/p, out/p, float64(ms)/1000/float64(p))
	}

	for _, r := range reps {
		fmt.Fprintf(w, "\n## `%s`: top confusions\n\n| Labeled | Predicted | Rows / pass |\n|---|---|---|\n", r.Name)
		for _, c := range r.Confusions[:min(8, len(r.Confusions))] {
			fmt.Fprintf(w, "| `%s` | `%s` | %.1f |\n", c.Want, c.Got, c.N)
		}
		fmt.Fprintf(w, "\nMost confident misses:\n\n| Payee | Labeled | Predicted | Confidence |\n|---|---|---|---|\n")
		for _, m := range r.Missed[:min(10, len(r.Missed))] {
			fmt.Fprintf(w, "| %s | `%s` | `%s` | %.2f |\n", m.Payee, m.Want, m.Got, m.Conf)
		}
	}
}
