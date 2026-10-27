"""URL sanitising/feature extraction, text normalisation, rule baseline and safe explanations.
NOTHING in this project opens, fetches or executes a URL (NFR-01) - links are only text."""
import re
from config import ABSTAIN_LOW, ABSTAIN_HIGH, MIN_WORDS

URL_RE = re.compile(r"(?i)\b(?:https?|hxxps?|h\*\*ps?)://[^\s<>\"']+")
SHORTENERS = {"short", "shrt", "lnk"}
CRED_PATH = ("login", "verify", "signin", "sign-in", "secure", "confirm", "claim", "pay", "update")


def refang(s: str) -> str:
    for a in ("[.]", "(.)", "(dot)", "[dot]"):
        s = s.replace(a, ".")
    return s


def defang_live(text: str) -> str:
    """Make every URL inert: http->hxxp and '.'->'[.]' in the host."""
    def fix(m):
        u = re.sub(r"(?i)^http(s?)://", lambda x: "hxxp" + x.group(1) + "://", m.group(0))
        scheme, rest = u.split("://", 1)
        host, sep, path = rest.partition("/")
        host = host.replace("[.]", ".").replace(".", "[.]")
        return f"{scheme}://{host}{sep}{path}"
    return URL_RE.sub(fix, text)


def url_features(text: str) -> dict:
    urls = URL_RE.findall(text)
    f = {"n_urls": len(urls), "shortener": 0, "host_hyphens": 0, "host_digits": 0,
         "cred_path": 0, "defanged": 0, "live_form": 0}
    for raw in urls:
        f["defanged"] |= int(bool(re.search(r"(?i)hxxp|h\*\*p|\[\.\]|\(dot\)", raw)))
        f["live_form"] |= int(bool(re.match(r"(?i)https?://", raw)) and "[.]" not in raw)
        u = re.sub(r"(?i)^(?:https?|hxxps?|h\*\*ps?)://", "", refang(raw))
        host, _, path = u.partition("/")
        host = host.lower().rstrip(".,;)")
        f["shortener"] |= int(host.split(".")[0] in SHORTENERS)
        f["host_hyphens"] = max(f["host_hyphens"], host.count("-"))
        f["host_digits"] = max(f["host_digits"], sum(c.isdigit() for c in host))
        f["cred_path"] |= int(any(k in path.lower() for k in CRED_PATH))
    return f


def feature_string(f: dict) -> str:
    return ";".join(f"{k}={v}" for k, v in f.items())


def text_for_model(subject: str, body: str) -> str:
    """Model input: text with URLs removed + safe tokens describing the URL and style."""
    f = url_features(body)
    plain = URL_RE.sub(" ", body)
    toks = ["urlpresent" if f["n_urls"] else "nourl"]
    if f["shortener"]: toks.append("urlshortener")
    if f["host_hyphens"] >= 2: toks.append("urlhyphen2")
    elif f["host_hyphens"] == 1: toks.append("urlhyphen1")
    if f["cred_path"]: toks.append("urlcredpath")
    if f["host_digits"]: toks.append("urldigits")
    full = f"{subject} {plain}"
    if full.count("!") >= 2: toks.append("manyexcl")
    letters = [c for c in full if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.3: toks.append("shouting")
    return (full + " " + " ".join(toks)).lower()


def normalize_for_dedup(text: str) -> str:
    text = URL_RE.sub(" url ", text.lower())
    text = re.sub(r"\d+", " n ", text)
    return re.sub(r"[^a-z\s]", " ", text)


# ---------------- rule baseline (FR-02) ----------------
RULES = [
    ("urgency", r"\b(urgent|immediately|final warning|final reminder|asap|expires?|last chance|permanent|within \d+ hours|before noon)\b", 1,
     "Uses urgency or pressure language"),
    ("credentials", r"\b(password|passwords|log ?in|sign ?in|credentials|username|verify|validate|revalidated|confirm your)\b", 1,
     "Asks about passwords, logins or identity checks"),
    ("money", r"\b(card details|bank details|direct deposit|gift cards?|vouchers?|pay now|payment failed|penalt\w+|legal action|settle|outstanding balance|codes)\b", 1,
     "Mentions payments, bank details or gift cards"),
    ("reward", r"\b(won|winner|prize|free|reward|congratulations|lucky)\b", 1,
     "Promises a prize or reward"),
    ("generic_greeting", r"\b(dear (customer|user|winner|employee)|attention)\b", 1,
     "Uses a generic greeting instead of your name"),
    ("secrecy", r"\b(between us|do not call|discreet|quiet favou?r|confidential)\b", 1,
     "Asks for secrecy"),
    ("safe_language", r"\b(no (further )?action (is )?(needed|required)|never ask|never share|if (this )?was you)\b", -1,
     "States no action is needed or warns against sharing secrets (reassuring)"),
    ("known_contact", r"\b(call the (number|helpdesk)|ext\.? ?\d+|helpdesk|hr desk)\b", -1,
     "Points to a known phone/desk contact rather than a link (reassuring)"),
]
RULE_THRESHOLD = 3


def rule_score(subject: str, body: str, sender_type: str = "unknown"):
    text = f"{subject} {URL_RE.sub(' ', body)}".lower()
    score, hits = 0, []
    for name, rx, w, why in RULES:
        if re.search(rx, text):
            score += w; hits.append(why)
    f = url_features(body)
    if f["shortener"]: score += 2; hits.append("Contains a shortened link that hides the destination")
    if f["host_hyphens"] >= 2 or f["cred_path"]:
        score += 1; hits.append("Link address looks unusual (many hyphens or login/verify path)")
    if sender_type in ("lookalike", "freemail", "unknown_external"):
        score += 1; hits.append("Sender domain is not an internal or known partner domain")
    elif sender_type == "internal":
        score -= 1; hits.append("Sender domain is internal (reassuring)")
    return score, hits


def rule_predict(df):
    sc = [rule_score(r.subject_or_title, r.body_text, r.sender_domain_type)[0] for r in df.itertuples()]
    return ([int(s >= RULE_THRESHOLD) for s in sc]), sc


# ---------------- abstain + safe explanation (FR-04) ----------------
def decide(p: float, body: str, subject: str = "") -> str:
    if len((subject + " " + body).split()) < MIN_WORDS:
        return "review"
    if p >= ABSTAIN_HIGH: return "phishing"
    if p <= ABSTAIN_LOW: return "legitimate"
    return "review"


_FRIENDLY = {"urlshortener": "shortened link", "urlhyphen2": "link host with many hyphens",
             "urlhyphen1": "link host with a hyphen", "urlcredpath": "link path asks to login/verify",
             "urlpresent": "contains a link", "nourl": "no link", "manyexcl": "many exclamation marks",
             "shouting": "ALL-CAPS style", "urldigits": "digits in link host"}


def explain(subject, body, sender_type, lr_pipeline, topk=5):
    """Safe explanation: rule cues + the words that pushed the linear model most. Never echoes URLs."""
    _, hits = rule_score(subject, body, sender_type)
    vec, clf = lr_pipeline.named_steps["tfidf"], lr_pipeline.named_steps["clf"]
    x = vec.transform([text_for_model(subject, body)])
    names = vec.get_feature_names_out()
    contrib = x.multiply(clf.coef_[0]).tocoo()
    items = sorted(zip(contrib.col, contrib.data), key=lambda t: -abs(t[1]))
    up, down = [], []
    for c, v in items:
        t = " ".join(_FRIENDLY.get(w, w) for w in names[c].split())
        if t in up or t in down:
            continue
        if v > 0 and len(up) < topk: up.append(t)
        if v < 0 and len(down) < topk: down.append(t)
    return {"rule_cues": hits, "pushes_toward_phishing": up, "pushes_toward_legitimate": down}
