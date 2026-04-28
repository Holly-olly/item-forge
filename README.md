# item-forge

**LLM-based psychological scale item generation and in-silico validation** — 4 models × 3 retrieval conditions, dual psychometric validation (EFA + EGA), 89-item expert review.

---

## Key Insights

👉 LLMs can generate structurally valid item pools
👉 Item quality is driven by semantic alignment, not surface features
👉 Retrieval is model-dependent, not universally beneficial
👉 Construct asymmetry matters: one subscale (PE) is stable, second one (CI) is ambiguous
👉 Embedding-based metrics approximate expert judgment

---

## Results: LLM evaluation

Five of twelve conditions passed both EFA and EGA independently, meaning the generated item pools showed clear two-factor structure from two different analytic angles:

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

### Key findings

**The value of retrieval is model-dependent.**
GPT-4o mini performs best *without* retrieval (No-RAG: NMI 71.1%). Adding re-ranked context drops its NMI to 52.2% — a meaningful deterioration. Llama 3 shows the opposite: its best result is Re-Ranked RAG (NMI 69.2%). A retrieval pipeline is not universally beneficial.

**Gemini is the consistent outlier.**
Gemini is the weakest performer — EGA NMI 30.3% and EFA DAAL 50.7% in the RAG condition, the lowest across all tested models. This may reflect sensitivity to two-factor prompt framing — an open question for follow-up.

**EFA and EGA disagree more than expected.**
Of 12 conditions, only 5 produce agreement between methods. Factor structure (EFA) and network community stability (EGA) capture different things: pools that look structurally clean in embedding space can still produce unstable communities, and vice versa. The dual-method filter is more conservative — and more defensible — than either method alone.

---

## Human evaluation

89 items — drawn from three of the five passing conditions (GPT-4o mini RAG, DeepSeek No-RAG, Llama 3 No-RAG) — entered blind expert review. Three domain experts rated each item on a 6-option scale (CI Keep / CI Revise / PE Keep / PE Revise / Both / Neither), with each item rated by exactly 2 experts.

### Key findings

**Cosine similarity to the PE definition is the strongest predictor of whether an item is correctly classified, disagreed on, or rejected** — rater identity explains almost nothing. Items fail expert review for semantic reasons, not writing quality: readability, sentiment, and word length predict rejection at near-chance level (AUC ≈ 0.50), while cosine similarity to the construct definition alone reaches AUC = 0.68. This suggests embedding-based semantic measures can serve as a useful proxy for preliminary item screening.

**PE is more consistently identified than CI** in both embedding space and expert ratings. This may reflect sharper semantic boundaries around PE, but definition wording and embedding bias cannot be ruled out as contributing factors. The CI/PE boundary is asymmetric: items anchored near PE are reliably classified and retained; CI-anchored items produce higher disagreement and more Neither ratings.

**Inter-rater agreement is fair to moderate** (mean κ = 0.35; range 0.22–0.44), below the threshold for substantive conclusions (Krippendorff α ≥ 0.667). Agreement patterns are consistent with item-level ambiguity rather than rater-specific tendencies — item content predicts agreement better than rater identity — but neither effect reached statistical significance (n = 89). Expert labels should be treated as noisy signal rather than definitive ground truth.

**Expert alignment with intended subscale**

| Expert | Match | Both | Neither | Mismatch |
|--------|-------|------|---------|----------|
| E1 | 78% | 8% | 12% | 2% |
| E2 | 75% | 12% | 12% | 1% |
| E3 | 71% | 12% | 15% | 2% |


**Include rate by model**
The percentage of items from that model that both raters agreed on the same subscale (CI or PE)

| Model | Included | Total | Include rate* |
|-------|----------|-------|------|
| DeepSeek | 25 | 36 | 69% |
| GPT-4o mini | 15 | 27 | 56% |
| Llama 3 | 10 | 26 | 38% |


### Next steps

The next phase of the project focuses on **improving measurement quality before expanding data collection**.

Rather than relying on additional human annotation at the current stage, the pipeline will be extended with **an iterative generation and pre-screening loop**, where candidate items are evaluated automatically prior to expert review. This stage will combine semantic alignment measures (e.g., embedding-based similarity to construct definitions) with structural signals to identify items that are both construct-relevant and well-separated at the factor level.

A key objective is to refine construct representation, particularly for Consistency of Interest (CI), which shows systematic ambiguity in both embedding space and human judgment. This involves restructuring how constructs are specified and operationalized during generation, rather than only tuning model parameters.

Human evaluation will then be reintroduced on a higher-quality, pre-filtered item pool, enabling more reliable agreement and more efficient use of expert input. Participant-based validation will follow once item stability and construct clarity are established.

This approach positions the framework not only as a generation pipeline, but as a **scalable system for construct-driven item development**, where automated semantic evaluation reduces reliance on early-stage human screening while preserving psychometric rigor.

---

## Why Grit is a hard test case

| Subscale | Definition |
|---|---|
| **Consistency of Interest (CI)** | Maintaining focus on a single long-term ambition; resisting distraction from new interests |
| **Perseverance of Effort (PE)** | Sustaining hard work through setbacks; finishing what is started |

The two-factor structure is actively debated — some analyses favour a unidimensional solution, particularly when CI loadings are weak across samples (Credé et al., 2017). A pipeline that generates discriminating CI items is doing something non-trivial. The asymmetry found in human review (PE reliably identified, CI ambiguous) mirrors this wider measurement challenge.

The cosine similarity between the CI and PE construct definitions is 0.41, indicating moderate semantic overlap in embedding space — which sets a theoretical floor on how well any embedding-based method can discriminate between the two subscales.

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
       40 items per subscale (~80 per condition)
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

**Fixed prompt.** All 12 conditions use the same prompt. Model-specific tuning — persona framing, few-shot examples, output format constraints — is the most immediate lever for improvement.

**Fixed temperature.** Lower values may reduce semantic diffuseness in models that produced cross-loading items.

**Unidimensional framing untested.** Items were generated against the two-factor model throughout. Whether prompting for Grit as a unified construct produces better or worse pools is an open question.

**Tiebreaks pending.** 16 of 89 items remain unresolved — subscale conflict between two raters requires a third independent rating before final retention decisions.

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
