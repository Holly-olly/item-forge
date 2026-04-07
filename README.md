# item-forge

**LLM-based psychological scale item generation and in-silico validation**

A construct-agnostic pipeline for generating and psychometrically validating psychological scale items using large language models. Demonstrated on the Grit scale (Duckworth et al., 2007).

---

## What this is

Scale development typically requires domain experts to write items by hand, with iteration cycles measured in months. This pipeline tests whether LLMs can replace or augment that process — and if so, **which models, under which retrieval conditions, produce psychometrically usable items**.

**4 models × 3 retrieval conditions = 12 pools**, each validated independently by two psychometric methods before items reach human review.

---

## Results

### Which conditions passed validation?

Five of twelve conditions passed both EFA and EGA independently — meaning the generated item pools showed clear two-factor structure from two different analytic angles:

| Model | Condition | EFA (DAAL%) | EGA (NMI%) | Verdict |
|---|---|---|---|---|
| GPT-4o mini | No-RAG | 92.9% ✓ | 71.1% ✓ | ✅ Both pass |
| GPT-4o mini | RAG | 92.1% ✓ | 67.7% ✓ | ✅ Both pass |
| DeepSeek | RAG | 97.0% ✓ | 67.3% ✓ | ✅ Both pass |
| Llama 3 | RAG | 98.5% ✓ | 67.1% ✓ | ✅ Both pass |
| Llama 3 | Re-Ranked RAG | 100.0% ✓ | 69.2% ✓ | ✅ Both pass |
| Gemini | RAG | 50.7% ✗ | 30.3% ✗ | ✗ Both fail |
| GPT-4o mini | Re-Ranked RAG | 54.8% ✗ | 52.2% ✗ | ✗ Both fail |

Full results for all 12 conditions: [`outputs/table1_master_results.csv`](outputs/table1_master_results.csv) · [`outputs/table2_method_agreement.csv`](outputs/table2_method_agreement.csv)

### Three findings worth noting

**1. RAG helps DeepSeek and Llama; it hurts GPT and Gemini.**
GPT performs best *without* retrieval (No-RAG: NMI 71.1%). Adding re-ranked context drops its NMI to 52.2% — a meaningful deterioration. Llama shows the opposite pattern: its best result is Re-Ranked RAG (NMI 69.2%). This means the value of a retrieval pipeline is model-dependent, not universal.

**2. Gemini is the consistent outlier.**
Across all three conditions, Gemini produces the weakest EGA structure (NMI: 67.1 → 30.3 → 44.3). Its EFA pass rate in No-RAG is also the lowest (48.7%). This may reflect sensitivity to the two-factor prompt framing — an open question for follow-up.

**3. The two methods disagree more than expected.**
Of 12 conditions, only 5 see agreement. The EFA–EGA split reveals that factor structure (EFA) and network community stability (EGA) capture different things: some pools look structurally clean in embedding space but produce unstable communities, and vice versa. This makes the dual-method filter more conservative — and more defensible — than either method alone.

### Final item pool

89 items passed both validation methods and entered human review: [`outputs/human_review_items.csv`](outputs/human_review_items.csv)

---

## Human evaluation design

The 89-item pool is undergoing content validity review by 3 domain experts using a blind factor assignment protocol.

**Format:** Items are shown without subscale labels. Each expert makes one decision per item:

| Option | Meaning |
|--------|---------|
| CI — Keep | Measures Consistency of Interest; include as-is |
| CI — Revise | Measures CI; wording needs improvement |
| PE — Keep | Measures Perseverance of Effort; include as-is |
| PE — Revise | Measures PE; wording needs improvement |
| Both | Measures Grit generally; does not discriminate CI from PE |
| Neither | Off-construct; remove |

**Why blind assignment:** showing experts the intended subscale anchors them toward confirmation. Blind assignment tests whether items are face-valid enough to be correctly categorised — consistent with 2025 CVR benchmarks (82% correct construct assignment reported as the validity standard).

**Why one question per item:** selecting CI or PE already encodes both the construct assignment and the quality rating (Keep vs Revise). Selecting Both or Neither is the complete decision — no quality rating is meaningful for items that fail the assignment test.

**Set split:** 89 items split into 3 balanced sets (~30 each, stratified by subscale and generation condition). Each expert rates 2 sets (~60 items, ~25 min). Every item receives exactly 2 independent ratings.

**Retention:** items kept if both raters assign the same subscale and at least one rates Keep or Revise. Items rated Both by both experts are preserved in a separate unidimensional pool for potential future use.

Full protocol: [`human_rate_report.md`](human_rate_report.md)

## Construct: Grit

| Subscale | Definition |
|---|---|
| **Consistency of Interest (CI)** | Maintaining focus on a single long-term ambition; resisting distraction from new interests |
| **Perseverance of Effort (PE)** | Sustaining hard work through setbacks; finishing what is started |

The two-factor structure is actively debated — some analyses favour a unidimensional solution, particularly when CI loadings are weak across samples (Credé et al., 2017). A pipeline that generates discriminating CI items is doing something non-trivial.

---

## Pipeline

```
Construct definition + factor structure
          │
          ▼
  Semantic reference index (FAISS)
  (Big Five, remote work, maladaptive,
   grit literature — 20 definitions)
          │
     ┌────┴────────┬──────────────────┐
     ▼             ▼                  ▼
  No-RAG         RAG          Re-Ranked RAG
     └────────────┴──────────────────┘
                  │
       4 LLMs × 3 conditions = 12 pools
       ~80 items generated per condition
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
  EFA (DAAL ≥ 80%)      EGA (NMI ≥ 65%)
  Embedding factor      Network community
  analysis              stability (bootEGA)
       └──────────┬──────────┘
                  ▼
     5 conditions pass both filters
          89 items → human review
```

**Models:** DeepSeek (Ollama local) · GPT-4o mini · Gemini 1.5 Pro · Llama 3 (Ollama local)  
**Embedding model:** `sentence-transformers/all-mpnet-base-v2` (768D) for reference index  
**Validation embedding:** `MiniLM-L12-v2` for EFA  
**Temperature:** 1.0 (fixed across all conditions)

---

## Limitations

**Fixed prompt.** All 12 conditions use the same prompt. Model-specific tuning — persona framing, few-shot examples, output format constraints — is the most immediate lever for improvement and could shift results substantially.

**Fixed temperature.** Lower values may reduce semantic diffuseness in models that produced cross-loading items. Temperature sensitivity is the next experiment.

**Unidimensional framing untested.** Items were generated against the two-factor model throughout. Whether prompting for Grit as a unified construct produces better or worse pools is an open question with direct relevance to the ongoing CI/PE dimensionality debate.

**Human review pending.** The 89-item pool has not yet been rated by domain experts. Psychometric properties reported here are in-silico only.

---

## How to run

```bash
git clone https://github.com/Holly-olly/item-forge
cd item-forge
pip install pandas numpy sentence-transformers scikit-learn matplotlib \
    openai google-genai groq ollama faiss-cpu python-dotenv \
    tqdm factor_analyzer ipywidgets

# Set API keys
cp .env.example .env
# Fill in: OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
# DeepSeek and Llama 3 run locally via Ollama:
# ollama pull deepseek-r1:8b && ollama pull llama3

jupyter lab item_forge.ipynb
```

To view results only — no generation required: run Phase 8–10 cells. Generation cells (Phases 4–6) can be skipped if `outputs/` already exists.

---

## Acknowledgements

**EFA validation** adapted from [mistr3ated/AI-Psychometrics-Nigel](https://github.com/mistr3ated/AI-Psychometrics-Nigel) — embedding-based pseudo factor analysis for psychometric item pool evaluation.

**AI-GENIE pipeline** adapted from [laralee/AIGENIE](https://github.com/laralee/AIGENIE) — AI-Generated Item Networks with EGA (Russell-Lasalandra, Christensen & Golino, 2025).

---

## References

Duckworth, A. L., Peterson, C., Matthews, M. D., & Kelly, D. R. (2007). Grit: Perseverance and passion for long-term goals. *Journal of Personality and Social Psychology, 92*(6), 1087–1101.

Credé, M., Tynan, M. C., & Harms, P. D. (2017). Much ado about grit: A meta-analytic synthesis of the grit literature. *Journal of Personality and Social Psychology, 113*(3), 492–511.

Russell-Lasalandra, M., Christensen, A. P., & Golino, H. (2025). AI-GENIE: AI-Generated Item Networks with Exploratory Graph Analysis. *PsyArXiv.* https://doi.org/10.31234/osf.io/XXXX
