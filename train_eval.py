import datetime as dt
import hashlib
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.pipeline import Pipeline

from config import *
from cues import decide, explain, rule_predict, text_for_model


def read(p): return pd.read_csv(p, keep_default_na=False)
def texts(df): return [text_for_model(s, b) for s, b in zip(df.subject_or_title, df.body_text)]


def build(name):
    clf = (LogisticRegression(max_iter=2000, C=3.0, class_weight="balanced", random_state=SEED) if name == "TF-IDF + Logistic Regression"
           else RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=-1))
    return Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)), ("clf", clf)])


def metrics(df, pred, name, split):
    y, pred = df.label.values, np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    um = (df.urgent_legit.values == 1) & (y == 0)
    return dict(model=name, eval_set=split, n=len(df), accuracy=accuracy_score(y, pred),
                precision=precision_score(y, pred, zero_division=0), recall=recall_score(y, pred, zero_division=0),
                f1=f1_score(y, pred, zero_division=0), fpr=fp / max(fp + tn, 1),
                fpr_legit_urgent=(pred[um].mean() if um.sum() else np.nan), n_legit_urgent=int(um.sum()),
                tp=tp, fp=fp, fn=fn, tn=tn)


def main():
    tr, te = read(DATA / "train.csv"), read(DATA / "test_campaign_holdout.csv")
    full = read(DATA / "deduped_messages.csv")
    Xtr, Xte, ytr, yte = texts(tr), texts(te), tr.label.values, te.label.values
    names = ["TF-IDF + Logistic Regression", "Random Forest (TF-IDF)"]
    rows, fitted, proba = [], {}, {}

    # ---------- baseline rules ----------
    rp, rs = rule_predict(te)
    rows.append(metrics(te, rp, "Rule baseline", "campaign holdout"))

    # ---------- leave-campaign-out CV on TRAIN only -> model selection (holdout untouched) ----------
    cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=SEED)
    cvres = {}
    for n in names:
        sc = []
        for a, b in cv.split(Xtr, ytr, tr.campaign_group):
            m = build(n).fit([Xtr[i] for i in a], ytr[a])
            sc.append(f1_score(ytr[b], m.predict([Xtr[i] for i in b]), zero_division=0))
        cvres[n] = float(np.mean(sc))
    best_name = max(cvres, key=cvres.get)

    # ---------- final fit + campaign holdout ----------
    for n in names:
        m = build(n).fit(Xtr, ytr)
        fitted[n] = m
        proba[n] = m.predict_proba(Xte)[:, 1]
        rows.append(metrics(te, (proba[n] >= .5).astype(int), n, "campaign holdout"))

    # ---------- comparison: random split on the same de-duplicated data (optimistic / leaky) ----------
    ftr, fte = train_test_split(full, test_size=.3, stratify=full.label, random_state=SEED)
    for n in names:
        m = build(n).fit(texts(ftr), ftr.label.values)
        rows.append(metrics(fte, m.predict(texts(fte)), n, "random split (optimistic)"))
    rows.append(metrics(fte, rule_predict(fte)[0], "Rule baseline", "random split (optimistic)"))

    res = pd.DataFrame(rows)
    res.round(4).to_csv(REPORTS / "results_table.csv", index=False)
    show = res[["model", "eval_set", "n", "accuracy", "precision", "recall", "f1", "fpr", "fpr_legit_urgent"]].round(3)
    (REPORTS / "results_table.md").write_text(show.to_markdown(index=False))
    print(show.to_string(index=False))
    print("\nLeave-campaign-out CV F1 (train only):", {k: round(v, 3) for k, v in cvres.items()}, "-> selected:", best_name)

    # ---------- confusion matrices ----------
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, (n, p) in zip(axs, [("Rule baseline", rp)] + [(n, (proba[n] >= .5).astype(int)) for n in names]):
        sns.heatmap(confusion_matrix(yte, p, labels=[0, 1]), annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["legit", "phish"], yticklabels=["legit", "phish"])
        ax.set_title(n, fontsize=9); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.suptitle("Confusion matrices - campaign holdout"); plt.tight_layout()
    plt.savefig(REPORTS / "confusion_matrices.png", dpi=130); plt.close()

    # ---------- abstain analysis (FR-04) + per-campaign errors ----------
    p = proba[best_name]
    dec = np.array([decide(pp, b, s) for pp, b, s in zip(p, te.body_text, te.subject_or_title)])
    auto = dec != "review"
    pred_auto = (dec == "phishing").astype(int)
    abst = dict(coverage=float(auto.mean()), abstain_rate=float(1 - auto.mean()),
                selective_accuracy=float((pred_auto[auto] == yte[auto]).mean()) if auto.any() else None,
                abstain_rate_phishing=float((~auto)[yte == 1].mean()), abstain_rate_legit=float((~auto)[yte == 0].mean()),
                thresholds=[ABSTAIN_LOW, ABSTAIN_HIGH])
    pc = te.assign(correct=((p >= .5).astype(int) == yte), review=~auto).groupby("campaign_group").agg(
        n=("label", "size"), label=("label", "first"), accuracy=("correct", "mean"), review_rate=("review", "mean")).round(3)
    pc.to_csv(REPORTS / "errors_by_campaign.csv")
    print("\nAbstain analysis:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in abst.items()})
    print("\nBy held-out campaign:\n", pc.to_string())

    # ---------- 3 error examples (T5 / CP-4) ----------
    err = te.assign(p=p, pred=(p >= .5).astype(int))
    fn = err[(err.label == 1) & (err.pred == 0)].sort_values("p").head(1)
    fpu = err[(err.label == 0) & (err.pred == 1) & (err.urgent_legit == 1)].sort_values("p", ascending=False).head(1)
    rest = err[(err.label != err.pred) & ~err.message_id.isin(list(fn.message_id) + list(fpu.message_id))].head(3)
    ex = pd.concat([fn, fpu, rest]).head(3)
    if len(ex) < 3:  # fewer than 3 errors -> add least-confident correct predictions
        ok = err[err.label == err.pred].assign(u=lambda d: (d.p - .5).abs()).sort_values("u").head(3 - len(ex))
        ex = pd.concat([ex, ok])
    lines = ["# Error examples (campaign holdout, best model)\n"]
    for r in ex.itertuples():
        e = explain(r.subject_or_title, r.body_text, r.sender_domain_type, fitted["TF-IDF + Logistic Regression"])
        kind = "missed phishing (false negative)" if r.label == 1 and r.pred == 0 else "legitimate flagged (false positive)" if r.label == 0 and r.pred == 1 else "correct but low confidence"
        lines.append(f"## {r.message_id} - {kind}\n* Campaign: `{r.campaign_group}` | true label: {'phishing' if r.label else 'legitimate'} | P(phishing) = {r.p:.2f}\n"
                     f"* Subject: {r.subject_or_title or '(SMS)'}\n* Body (defanged): {r.body_text!r}\n* Rule cues: {e['rule_cues']}\n"
                     f"* Words pushing toward phishing: {e['pushes_toward_phishing']}; toward legitimate: {e['pushes_toward_legitimate']}\n"
                     f"* Why it went wrong: see report discussion - the model has not seen this campaign's wording.\n")
    (REPORTS / "error_examples.md").write_text("\n".join(lines))

    # ---------- save versioned model ----------
    sha = hashlib.sha256((DATA / "train.csv").read_bytes()).hexdigest()[:16]
    bundle = dict(model=fitted[best_name], explainer=fitted["TF-IDF + Logistic Regression"], best_name=best_name,
                  version=MODEL_VERSION, thresholds=(ABSTAIN_LOW, ABSTAIN_HIGH), data_sha256=sha,
                  trained_at=dt.datetime.now().isoformat(timespec="seconds"))
    joblib.dump(bundle, MODELS / "phishshift_model.joblib")
    card = dict(version=MODEL_VERSION, selected=best_name, cv_f1_leave_campaign_out=cvres, data_sha256=sha, seed=SEED,
                holdout=res[res.eval_set == "campaign holdout"].round(4).to_dict("records"), abstain=abst,
                trained_at=bundle["trained_at"])
    json.dump(card, open(MODELS / "model_card.json", "w"), indent=2, default=float)
    with open(REPORTS / "experiment_log.jsonl", "a") as f:   # T7 experiment journal (append-only)
        f.write(json.dumps(dict(time=bundle["trained_at"], version=MODEL_VERSION, seed=SEED, dedup_threshold=DEDUP_THRESHOLD,
                                holdout=HOLDOUT_CAMPAIGNS, selected=best_name, cv=cvres, data_sha256=sha), default=float) + "\n")

    # ---------- optional mentor-supplied variant ----------
    mp = DATA / "mentor_holdout.csv"
    if mp.exists():
        from prepare_data import sanitize
        mdf, _ = sanitize(pd.read_csv(mp, keep_default_na=False))
        if "urgent_legit" not in mdf: mdf["urgent_legit"] = 0
        mr = [metrics(mdf, rule_predict(mdf)[0], "Rule baseline", "mentor holdout")] + \
             [metrics(mdf, fitted[n].predict(texts(mdf)), n, "mentor holdout") for n in names]
        pd.DataFrame(mr).round(4).to_csv(REPORTS / "mentor_holdout_results.csv", index=False)
        print("\nMentor holdout evaluated -> reports/mentor_holdout_results.csv")


if __name__ == "__main__":
    main()
