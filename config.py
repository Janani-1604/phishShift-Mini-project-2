"""Central settings. Change HOLDOUT_CAMPAIGNS when your mentor assigns a variant."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA, REPORTS, MODELS = ROOT / "data", ROOT / "reports", ROOT / "models"
for _p in (DATA, REPORTS, MODELS):
    _p.mkdir(exist_ok=True)

SEED = 42
DEDUP_THRESHOLD = 0.90          # cosine similarity above which two messages are "near-identical"
MODEL_VERSION = "phishshift-1.0.0"
ABSTAIN_LOW, ABSTAIN_HIGH = 0.35, 0.65   # p(phishing) between these -> "needs human review"
MIN_WORDS = 4                   # shorter messages are never auto-classified

# Whole campaigns kept OUT of training (FR-03). Includes unfamiliar-wording phishing
# and legitimate urgent messages that mention passwords / payroll.
HOLDOUT_CAMPAIGNS = ["shared_doc_soft", "payroll_portal_soft", "ceo_favour",
                     "pw_expiry_legit", "payroll_calendar_legit",
                     "bank_alert_legit", "security_training_legit"]

SCHEMA = ["message_id", "channel", "subject_or_title", "body_text", "sanitized_url_features",
          "sender_domain_type", "label", "campaign_group", "collected_date"]
