# Remote Work Scale — Psychometric Item Generation Lab (2026)

## Overview
Generates and validates survey items for two remote-work personality scales using RAG-augmented LLMs.

## Scales
- **Adaptability** — ability and motivation to change or fit different task, social, or environmental features
- **Exploratoriness** — individual differences in the tendency to seek out new experiences and explore ideas, values, emotions, and sensations

## Pipeline
1. **Phase 1** — Semantic heatmaps (convergent/discriminant validity vs. Big 5 and Maladaptive traits)
2. **Phase 2** — IPIP item embedding + FAISS index (3805 items, 1536D)
3. **Phase 3** — Retrieval validation
4. **Phases 4–6** — Item generation: No-RAG | RAG | Re-ranked RAG × 4 LLMs
5. **Phase 7** — Master CSV merge + cosine similarity quality evaluation

## Models
| Model | Provider |
|---|---|
| gemini-2.0-flash-lite | Google AI Studio |
| gpt-4o-mini | OpenAI |
| llama-3.3-70b-versatile | Groq |
| deepseek-r1:8b | Ollama (local) |

## Setup
1. Copy `.env.example` to `.env` and fill in your API keys
2. Install dependencies: `pip install pandas sentence-transformers scikit-learn matplotlib openai google-genai groq ollama faiss-cpu python-dotenv tqdm`
3. Ensure Ollama is running locally with `deepseek-r1:8b` pulled
4. Run `remote_work_scale.ipynb`
