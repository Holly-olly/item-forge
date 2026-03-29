# item-forge

> LLM-based psychological scale item generation and in-silico validation.

## Overview

Generates and validates survey items for two Grit personality scales using RAG-augmented LLMs,
then applies dual psychometric validation (EFA-based + AI-GENIE network psychometrics).
940 items generated across 12 conditions; 89 high-quality items exported for human expert review.

## Scales

| Scale | Description |
|---|---|
| **Consistency of Interest (CI)** | Maintaining focus on a single ambition; resisting the urge to shift to new projects |
| **Perseverance of Effort (PE)** | Sustaining hard work through setbacks; finishing initiated tasks despite difficulties |

## Pipeline

| Phase | Description |
|---|---|
| **1** | Semantic heatmaps: Grit vs Big 5, Remote vs Maladaptive — establish convergent and discriminant validity |
| **2** | IPIP item embedding (text-embedding-3-small) + FAISS index (3,805 items, 1536D) |
| **3** | Retrieval validation — verify FAISS returns semantically relevant IPIP items |
| **4–6** | Item generation: No-RAG / RAG / Re-ranked RAG × 4 LLMs (40 items/scale, temperature=1.0) |
| **7** | Master CSV merge + cosine similarity quality evaluation |
| **8** | EFA-based validation: pseudo-factor analysis on MiniLM-L12-v2 embeddings, DAAL accuracy, discrimination scores, cross-loading removal |
| **9** | AI-GENIE validation: EGA → UVA → bootEGA pipeline (R/EGAnet), NMI and community-match metrics |
| **10** | Results consolidation: master comparison table, method agreement analysis, final item pool for human review |

## Models

| Model | Provider | Temperature |
|---|---|---|
| gemini-2.0-flash-lite | Google AI Studio | 1.0 |
| gpt-4o-mini | OpenAI | 1.0 |
| llama-3.3-70b-versatile | Groq | 1.0 |
| deepseek-r1:8b | Ollama (local) | 1.0 |

## Key Results (Phase 10)

- **Best EFA-based validation:** DeepSeek — DAAL 94–99% across all conditions, highest discrimination scores
- **Best AI-GENIE validation:** Llama × No-RAG — NMI final 74.8%, 100% community-to-theory match
- **Method agreement:** 7/12 conditions agreed (5 both-pass, 2 both-fail)
- **Final human review pool:** 89 items (50 CI, 39 PE) from DeepSeek × No-RAG, GPT × RAG, Llama × No-RAG

See `outputs/human_review_items.csv` for the final item set and `outputs/table1_master_results.csv` for the full comparison table.

## Repository Structure

```
item_forge.ipynb        # Main notebook (Phases 1–10)
efa_items.py                   # Standalone EFA-based validation script
aigenie_items.R                # Standalone AI-GENIE pipeline (R)
aipsych_remote.csv             # Reference items for semantic heatmaps
TedoneItemAssignmentTable30APR21.csv  # IPIP item pool (Tedone, 2021)
items_*.csv                    # Generated items per model × condition (12 files)
outputs/
  final_item_bank.csv          # All items surviving EFA filtering (846 items)
  final_bank_summary.csv       # Per-condition EFA quality summary
  human_review_items.csv       # Final 89-item human review pool
  table1_master_results.csv    # Master validation table (12 conditions × all metrics)
  table2_method_agreement.csv  # EFA vs AI-GENIE method agreement per condition
  table3_human_review_counts.csv
  table4_top10_items.csv
```

## Setup

1. Copy `.env.example` to `.env` and fill in your API keys (OpenAI, Google, Groq):
   ```
   cp .env.example .env
   ```
2. Install Python dependencies:
   ```
   pip install pandas numpy sentence-transformers scikit-learn matplotlib \
               openai google-genai groq ollama faiss-cpu python-dotenv \
               tqdm factor_analyzer ipywidgets
   ```
3. Install R dependencies (for Phase 9 — AI-GENIE validation):
   ```r
   install.packages(c("EGAnet", "igraph"))
   ```
4. Ensure Ollama is running locally with `deepseek-r1:8b` pulled:
   ```
   ollama pull deepseek-r1:8b
   ```
5. Run `item_forge.ipynb` sequentially from Phase 1.

## Standalone Scripts

Both scripts can be run independently on any items CSV:

```bash
# EFA-based validation
python efa_items.py --items my_items.csv --output-dir results/

# AI-GENIE validation (requires embeddings produced by efa_items.py)
Rscript aigenie_items.R \
    --items my_items.csv \
    --embeddings results/embeddings.csv \
    --output-dir results/ \
    --boot-iter 100 \
    --seed 42
```

## Acknowledgements

This project builds on two open-source works, both released under the MIT License:

**EFA-based validation approach**
Adapted from [mistr3ated/AI-Psychometrics-Nigel](https://github.com/mistr3ated/AI-Psychometrics-Nigel)
— embedding-based pseudo factor analysis for psychometric item pool evaluation.
([License](https://github.com/mistr3ated/AI-Psychometrics-Nigel/blob/main/LICENSE))

**AI-GENIE pipeline**
Adapted from [laralee/AIGENIE](https://github.com/laralee/AIGENIE)
— AI-Generated Item Networks with EGA: network psychometrics pipeline for item pool validation
(Russell-Lasalandra, Christensen & Golino, 2025).
([License](https://github.com/laralee/AIGENIE/blob/main/LICENSE))

## References

Duckworth, A. L., Peterson, C., Matthews, M. D., & Kelly, D. R. (2007).
Grit: Perseverance and passion for long-term goals.
*Journal of Personality and Social Psychology, 92*(6), 1087–1101.

Goldberg, L. R. (1999). A broad-bandwidth, public-domain, personality inventory measuring
the lower-level facets of several five-factor models. *Personality Psychology in Europe, 7*, 7–28.
([IPIP](https://ipip.ori.org))

Russell-Lasalandra, M., Christensen, A. P., & Golino, H. (2025).
AI-GENIE: AI-Generated Item Networks with Exploratory Graph Analysis.
*PsyArXiv.* https://doi.org/10.31234/osf.io/XXXX

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
