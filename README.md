# item-forge

**LLM-based psychological scale item generation and in-silico validation**

A construct-agnostic pipeline for generating and psychometrically validating psychological
scale items using large language models. Demonstrated on the Grit scale (Duckworth et al., 2007).

---

## What this is

Scale development typically requires domain experts to write items by hand, with iteration
cycles measured in months. This pipeline tests whether LLMs can replace or augment that
process — and if so, which models, under which retrieval conditions, and with what
measurable psychometric quality.

**4 models × 3 retrieval conditions = 12 pools**, evaluated under fixed prompt and
temperature (T = 1.0). Validated by two independent psychometric methods.

---

## Key findings

![Model × condition heatmap](outputs/figures/chart_model_condition_heatmap.png)

- **RAG does not uniformly help.** Some models perform equally well with no retrieval;
  others improve substantially. Knowing which is which has direct practical value when
  API costs are a constraint.
- **CI items remain the hard case.** Consistency of Interest discrimination is the
  binding constraint across all 12 conditions. PE items are consistently easier
  to generate cleanly.
- **Cross-method agreement (EFA × EGA)** is the most conservative filter.
  The 89-item final pool passed both methods independently.

Full results by model × condition: see [`item_forge.ipynb`](item_forge.ipynb) Phase 10.

---

## Construct: Grit

| Subscale | Definition |
|---|---|
| **Consistency of Interest (CI)** | Maintaining focus on a single long-term ambition; resisting distraction |
| **Perseverance of Effort (PE)** | Sustaining hard work through setbacks; finishing what is started |

CI items are the hard case. The two-factor structure is debated — some analyses favour
a unidimensional solution, particularly when CI loadings are weak. A pipeline that
produces discriminating CI items is doing something non-trivial.

---

## Pipeline

```
Construct definition + factor structure
          │
          ▼
  Semantic reference index (FAISS)
          │
     ┌────┴────────┬──────────────────┐
     ▼             ▼                  ▼
  No-RAG         RAG          Re-ranked RAG
     └────────────┴──────────────────┘
                  │
       4 LLMs × 3 conditions = 12 pools
                  │
       ┌──────────┴──────────┐
       ▼                     ▼
  Phase 8: EFA          Phase 9: EGA
  (MiniLM-L12-v2)       (network psychometrics)
       └──────────┬──────────┘
                  ▼
     Phase 10: cross-method comparison
          89 items → human review
```

**Models:** DeepSeek · GPT-4o mini · Gemini · Llama 3
**Retrieval:** No-RAG · RAG · Re-ranked RAG
**Validation:** EFA on sentence embeddings + EGA / bootEGA

---

## Limitations and open questions

**Fixed prompt.** All 12 conditions use the same prompt. Model-specific tuning is
the most immediate lever for improvement.

**Fixed temperature (T = 1.0).** Temperature effects on item diversity and cross-loading
are the next experiment.

**Unidimensional framing untested.** Whether prompting for Grit as a unified construct
produces better or worse pools remains open.

**Human review pending.** The 89-item pool has not yet been rated by domain experts.

---

## How to run

```bash
git clone https://github.com/Holly-olly/item-forge
cd item-forge
pip install pandas numpy sentence-transformers scikit-learn matplotlib \
            openai google-genai groq ollama faiss-cpu python-dotenv \
            tqdm factor_analyzer ipywidgets
# Copy .env.example to .env and fill in API keys: OPENAI_API_KEY, GEMINI_API_KEY, GROQ_API_KEY
# DeepSeek runs locally via Ollama: ollama pull deepseek-r1:8b
cp .env.example .env
jupyter lab item_forge.ipynb
```

To view results only (no generation): run Phase 8–10 cells and the Results section.
Generation cells (Phases 4–6) can be skipped if `outputs/` already exists.

---

## Acknowledgements

**EFA-based validation approach**
Adapted from [mistr3ated/AI-Psychometrics-Nigel](https://github.com/mistr3ated/AI-Psychometrics-Nigel)
— embedding-based pseudo factor analysis for psychometric item pool evaluation.
([License](https://github.com/mistr3ated/AI-Psychometrics-Nigel/blob/main/LICENSE))

**AI-GENIE pipeline**
Adapted from [laralee/AIGENIE](https://github.com/laralee/AIGENIE)
— AI-Generated Item Networks with EGA: network psychometrics pipeline for item pool validation
(Russell-Lasalandra, Christensen & Golino, 2025).
([License](https://github.com/laralee/AIGENIE/blob/main/LICENSE))

---

## References

Duckworth, A. L., Peterson, C., Matthews, M. D., & Kelly, D. R. (2007).
Grit: Perseverance and passion for long-term goals.
*Journal of Personality and Social Psychology, 92*(6), 1087–1101.

Credé, M., Tynan, M. C., & Harms, P. D. (2017). Much ado about grit:
A meta-analytic synthesis of the grit literature.
*Journal of Personality and Social Psychology, 113*(3), 492–511.

Goldberg, L. R. (1999). A broad-bandwidth, public-domain, personality inventory measuring
the lower-level facets of several five-factor models. *Personality Psychology in Europe, 7*, 7–28.
([IPIP](https://ipip.ori.org))

Russell-Lasalandra, M., Christensen, A. P., & Golino, H. (2025).
AI-GENIE: AI-Generated Item Networks with Exploratory Graph Analysis.
*PsyArXiv.* https://doi.org/10.31234/osf.io/XXXX
