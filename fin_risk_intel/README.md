# AI & ML Powered Strategic Financial Systems - governed risk-intelligence prototype
BITS Pilani WILP · S1-26_AIMLCZG628T · Dhinesh Raghunathan (2024AD05461) · runs fully offline on a laptop, all data synthetic.

## 1. Run it in VS Code (5 minutes)
1. Install Python 3.10-3.12 and VS Code (with the *Python* extension). Open this folder (`File > Open Folder`).
2. Terminal > New Terminal, then:
   - **Windows:** `py -m venv .venv` → `.venv\Scripts\activate` → `pip install -r requirements.txt`
   - **macOS/Linux:** `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
3. (Optional) `python run_all.py` – regenerates everything and prints the metrics and H1-H6 verdicts (~10 s).
4. **Demo UI:** `streamlit run app.py` (or Run and Debug → *2) Launch demo UI*). Browser opens at http://localhost:8501. First launch builds and caches the state (~15 s).
5. To reset: delete `artifacts/state.joblib` (or run `python run_all.py`). Everything is seeded (`config.SEED`) so results are reproducible.

## 2. What is in the box (mapped to the outline)
| Outline item | Where |
|---|---|
| 3.2 canonical data model (customers, accounts, documents, devices, transactions, beneficiaries, alerts, outcomes) | `src/data_gen.py` → `data/*.csv`, `data/data_dictionary.csv`, `artifacts/alerts.csv` |
| Onboarding risk, calibrated + explainable | `src/detection.py` (calibrated logistic regression, reason codes) |
| Rule, supervised, anomaly, graph baselines (4.4) | `src/detection.py` (7 rules; calibrated gradient boosting; Isolation Forest + autoencoder; peer-group deviation), `src/features.py` (graph features, Louvain communities, PageRank) |
| NLP + document AI (4.4) | `src/nlp_docai.py` (KYC image rendering, layout-region extraction, document reconciliation, NER, case-note classifier) |
| IR design (4.5) | `src/retrieval.py` (section chunks, BM25, dense, RRF fusion, reranker, version/authorisation filter, Recall@k, MRR, nDCG, 38-query benchmark) |
| Multimodal RAG with page citations | `src/retrieval.py` image/table chunks + page renders; shown in UI tab 5/6 |
| LLM copilot (cited, uncertainty, refuses unsupported) | `src/copilot.py` (extractive by default; optional LLM; groundedness + citation-precision checks; low-confidence refusal; high-impact refusal) |
| Responsible AI (4.6) | `src/responsible_ai.py`: fairness gaps, PII masking, prompt-injection guard + red-team, model cards (`artifacts/model_cards/`), corpus inventory, audit trail, PSI drift |
| Evaluation (4.7) | time-based split, PR-AUC, precision/recall at review capacity, Brier + reliability curves, ablations, bootstrap CI (H1) |
| Hypotheses H1-H6 (3.3) | `artifacts/hypothesis_results.csv`, UI tab 9 |
| Tools (4.8) | Python, scikit-learn, NetworkX, Plotly, Streamlit (see section 4 for the substitutions) |

## 3. UI tabs for the professor
1 Overview (architecture, traceability, demo script) · 2 Account opening (document vs application, CDD risk, NER) · 3 Surveillance & fraud (scenarios, de-duplication, stream replay, alert queue) · 4 Graph intelligence · 5 **Investigator copilot** (score, reason codes, linked entities, cited evidence incl. scanned pages, narrative with uncertainty, human decision + audit log, refusal demo) · 6 Retrieval lab · 7 Model evaluation (review-capacity slider, ablations, PR/reliability curves) · 8 Responsible AI · 9 Hypotheses.

## 4. Be upfront in the viva: what is real, what is a stand-in
- **Data is synthetic** (3,000 customers, ~130k transactions, ~7% fraud with hard "stealth" cases and benign look-alikes). Numbers show method behaviour, not production accuracy. Outline scope (4.2) already excludes production data.
- **Dense retrieval** defaults to TF-IDF+LSA and the reranker to a lexical/bigram scorer so it runs offline. With the default setup **H3 is not supported** (BM25 is as good or better on this small corpus). Install `sentence-transformers` and set `USE_TRANSFORMERS=1` (MiniLM + cross-encoder) and re-run to test H3 properly.
- **Multimodal**: image pages are represented by visual descriptors in the index (stand-in for CLIP/LayoutLMv3 embeddings). The mechanism is real (image-only chunks are findable only with the multimodal index; H4) but not a trained vision model. **OCR** is simulated unless `pytesseract` + the Tesseract binary are installed.
- **Explainability** uses an interpretable surrogate for local reason codes and permutation importance globally (SHAP, Evidently, Fairlearn, Neo4j, Kafka, FAISS/pgvector, MLflow, LangChain from the outline are not required; the equivalent logic is implemented in-library and can be swapped in).
- **H5 handling time is a simulation** (`config.HANDLING`); the UI collects investigator ratings and decisions so you can report a small pilot instead.
- **H6**: FPR gaps are small but FNR gaps exceed the 0.10 threshold because each segment has few true positives; reported as *not supported*, not hidden.
- **Graph** is built transductively on all customers without labels; fraud labels are never used to build graph features.
- **LLM** is optional: `export ANTHROPIC_API_KEY=...` (and `LLM_MODEL`) then tick *Use LLM* in tab 5. Output goes through the same groundedness check.

## 5. Customising
`config.py`: review capacity, fairness threshold, retrieval confidence threshold, dataset size, handling-time assumptions. Add policy documents in `src/retrieval.py` (`SPEC`), rules in `src/detection.py` (`RULES`).
