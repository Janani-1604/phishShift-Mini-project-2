import json
import re

import joblib
import pandas as pd
import streamlit as st

from config import MODELS, REPORTS, ABSTAIN_LOW, ABSTAIN_HIGH
from cues import URL_RE, decide, defang_live, explain, rule_score, text_for_model

st.set_page_config(page_title="PhishShift", page_icon="🛡️", layout="centered")
DISCLAIMER = "⚠️ Classification is **decision support, not a guarantee of safety**. Never click links or share credentials because of this tool's output."

EXAMPLES = {
    "(choose an example)": ("email", "", "", "unknown"),
    "Legitimate password-reset notice": ("email", "Password change reminder", "Hello Asha,\nUrgent reminder: your password expires in 2 days. Change it on your company laptop; we never ask for passwords by email.\nIT Security", "internal"),
    "Phishing - account suspended": ("email", "URGENT: Your account has been suspended", "Dear customer,\nWe detected unusual activity. Verify your password and card details immediately: hxxps://secure-bankcorp-check[.]test/verify\nSecurity Team", "lookalike"),
    "Phishing - polite wording": ("email", "Please confirm your timesheet details", "Hello Meera,\nThe finance team asked everyone to re-enter their login details on the new timesheet page.\nhxxps://timesheet-acme[.]test/signin\nThanks, Admin", "unknown_external"),
    "Legitimate bank SMS": ("sms", "", "Bank alert: a purchase of 45 USD was made on your card ending 4321. If this was you, no action is needed.", "unknown"),
}


@st.cache_resource
def load_bundle():
    p = MODELS / "phishshift_model.joblib"
    return joblib.load(p) if p.exists() else None


bundle = load_bundle()
st.title("🛡️ PhishShift")
st.caption("Message risk scanner - educational triage tool (simulated training messages only)")
st.warning(DISCLAIMER)

if bundle is None:
    st.error("No trained model found. Run `python run_all.py` first, then restart the app.")
    st.stop()

tab1, tab2 = st.tabs(["Scan a message", "Model evidence"])
with tab1:
    ex = st.selectbox("Try an example", list(EXAMPLES))
    channel0, subj0, body0, sender0 = EXAMPLES[ex]
    channel = st.radio("Channel", ["email", "sms"], index=0 if channel0 == "email" else 1, horizontal=True, key=f"c{ex}")
    sender_opts = ["unknown", "internal", "partner", "freemail", "lookalike", "unknown_external"]
    sender = st.selectbox("Sender domain type (optional context)", sender_opts, index=sender_opts.index(sender0), key=f"s{ex}")
    subject = st.text_input("Subject / title", value=subj0, key=f"su{ex}", disabled=(channel == "sms"))
    body = st.text_area("Message body (use defanged links, e.g. hxxps://site[.]test/path)", value=body0, height=160, key=f"b{ex}")

    if st.button("Analyze", type="primary"):
        body_s = body.strip()
        if not body_s:
            st.warning("Please enter a message body.")
        elif len(body_s) > 5000:
            st.warning("Message too long (max 5,000 characters).")
        else:
            try:
                if URL_RE.search(body_s) and defang_live(body_s) != body_s:
                    st.info("A live-form link was detected and made inert. Links are never opened or fetched.")
                    body_s = defang_live(body_s)
                subject_s = "" if channel == "sms" else subject.strip()
                p = float(bundle["model"].predict_proba([text_for_model(subject_s, body_s)])[0][1])
                d = decide(p, body_s, subject_s)
                if d == "phishing":
                    st.error(f"🚨 Likely PHISHING - P(phishing) = {p:.0%}")
                elif d == "legitimate":
                    st.success(f"✅ Likely LEGITIMATE - P(phishing) = {p:.0%}")
                else:
                    st.warning(f"🤔 NOT SURE - needs human review (P(phishing) = {p:.0%}, abstain band {ABSTAIN_LOW:.0%}-{ABSTAIN_HIGH:.0%})")
                e = explain(subject_s, body_s, sender, bundle["explainer"])
                st.subheader("Why? (risk cues)")
                if e["rule_cues"]: st.markdown("\n".join(f"- {c}" for c in e["rule_cues"]))
                else: st.write("No standard risk cues triggered.")
                c1, c2 = st.columns(2)
                c1.markdown("**Words pushing toward phishing**\n\n" + (", ".join(e["pushes_toward_phishing"]) or "-"))
                c2.markdown("**Words pushing toward legitimate**\n\n" + (", ".join(e["pushes_toward_legitimate"]) or "-"))
                st.caption(f"Model: {bundle['best_name']} · version {bundle['version']} · trained {bundle['trained_at']}")
            except Exception as ex_:  # graceful error handling (T6)
                st.error(f"Could not analyze this message ({type(ex_).__name__}). Please check the input.")
    st.caption(DISCLAIMER)

with tab2:
    st.write("Results below are on **campaigns excluded from training**, not a random split.")
    for f, fn in [("results_table.csv", st.dataframe), ("robustness_drift.csv", st.dataframe)]:
        if (REPORTS / f).exists(): st.markdown(f"**{f}**"); fn(pd.read_csv(REPORTS / f))
    if (REPORTS / "confusion_matrices.png").exists(): st.image(str(REPORTS / "confusion_matrices.png"))
    if (MODELS / "model_card.json").exists(): st.json(json.load(open(MODELS / "model_card.json")), expanded=False)
