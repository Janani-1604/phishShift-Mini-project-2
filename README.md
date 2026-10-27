# PhishShift - Message Risk Scanner (ML-07)

Educational triage tool that classifies **simulated** emails/SMS as *likely phishing*, *likely legitimate* or
*needs human review (abstain)* and explains the risk cues. It never opens, fetches or executes links (NFR-01).

## Run in VS Code
```bash
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python run_all.py                # generate data -> sanitise -> dedup -> train -> robustness -> tests
streamlit run app.py             # UI at http://localhost:8501
```
`run_all.py` is fully reproducible (seed 42) and writes every evidence file to `reports/`.

## Pipeline
| Step | File | Output |
|---|---|---|
| T2 Data | `generate_data.py` | `data/raw_messages.csv` (never modified), `data/source_manifest.csv` |
| T2/T3 | `prepare_data.py` | sanitisation log, `dedup_proof.csv`, train/holdout split, EDA figures + `eda_summary.md` |
| T4/T5 | `train_eval.py` | `results_table.csv`, confusion matrices, error examples, `models/phishshift_model.joblib`, `model_card.json`, `experiment_log.jsonl` |
| T5 | `robustness.py` | `robustness_drift.csv`, `cp5_eight_messages.csv`, `acceptance_tests.md` |
| T6 | `app.py` | Streamlit risk-review UI |
| T7 | `tests/`, `docs/` | tests, traceability, threat model, provenance, AI-usage disclosure |

## Mentor-assigned variant (Section 08)
1. Put the mentor's synthetic campaign + legitimate business set in `data/mentor_holdout.csv` (same schema as `data/raw_messages.csv`) **or** edit `HOLDOUT_CAMPAIGNS` in `config.py`.
2. Run `python run_all.py` again - `reports/mentor_holdout_results.csv` is produced automatically.

## Deploy (free) - Streamlit Community Cloud
1. Run `python run_all.py` locally so `models/` and `reports/` exist, then push the whole folder to a **public GitHub repo**.
2. https://share.streamlit.io -> New app -> choose repo, branch `main`, main file `app.py` -> Deploy.
3. Put the live link in your final report and README.

## Important caveats
* The corpus is synthetic (template-composed), so absolute scores are not real-world numbers; the **gap between random-split and campaign-holdout results** is the finding.
* Output is decision support, not a guarantee of safety.
* See `docs/AI_USAGE.md` - you must be able to explain and reproduce every file.
