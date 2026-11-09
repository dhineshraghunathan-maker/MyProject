"""Document AI (layout-aware extraction + reconciliation) and NLP (NER, case-note classification)."""
import re
from difflib import SequenceMatcher
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import config as C


def _sim(a, b): return SequenceMatcher(None, str(a).lower(), str(b).lower()).ratio()


def reconcile(kyc):
    """Identity & document reconciliation: declared vs document fields -> doc_mismatch score."""
    k = kyc.copy()
    k["name_sim"] = [_sim(a, b) for a, b in zip(k.declared_name, k.doc_name)]
    k["dob_match"] = (k.declared_dob == k.doc_dob).astype(int)
    k["addr_sim"] = [_sim(a, b) for a, b in zip(k.declared_address, k.doc_address)]
    k["doc_mismatch"] = (1 - (.5 * k.name_sim + .2 * k.dob_match + .3 * k.addr_sim)).round(4)
    return k.set_index("customer_id")


def _font(sz):
    for n in ("DejaVuSans.ttf", "arial.ttf", "Arial.ttf"):
        try: return ImageFont.truetype(n, sz)
        except Exception: pass
    return ImageFont.load_default()


FIELDS = [("Name", "doc_name", "declared_name"), ("Date of birth", "doc_dob", "declared_dob"),
          ("Address", "doc_address", "declared_address"), ("Occupation / Profession", "doc_occupation", "declared_occupation"),
          ("Source of income", "doc_source_of_income", "declared_source_of_income"), ("Income range", "doc_income_range", "declared_income_range"),
          ("PEP", "doc_pep", "declared_pep"), ("ID type", "doc_id_type", "declared_id_type"), ("ID number", "doc_id_number", "declared_id_number")]


def render_kyc_image(cid, row):
    """Draw a synthetic KYC/ID page; red boxes mark fields that differ from the application."""
    path = C.IMG / f"kyc_{cid}.png"
    img = Image.new("RGB", (820, 380), "white"); d = ImageDraw.Draw(img); boxes = {}
    d.rectangle([5, 5, 814, 374], outline="black", width=2)
    d.text((30, 20), f"IDENTITY DOCUMENT / KYC FORM  (synthetic)   {cid}", fill="black", font=_font(18))
    for j, (lab, dc, dec) in enumerate(FIELDS):
        y = 70 + j * 70; bad = dec is not None and str(row[dc]).lower() != str(row[dec]).lower()
        d.text((30, y), lab, fill=(90, 90, 90), font=_font(15))
        d.rectangle([200, y - 6, 800, y + 34], outline=(200, 30, 30) if bad else (170, 170, 170), width=3 if bad else 1)
        d.text((210, y), str(row[dc]), fill="black", font=_font(18)); boxes[lab] = (200, y - 6, 800, y + 34)
    img.save(path); return str(path), boxes


def layout_extract(path, boxes, row):
    """Region-wise extraction. Uses Tesseract if installed, else stored text (simulated OCR)."""
    try:
        import pytesseract
        img = Image.open(path)
        return {l: pytesseract.image_to_string(img.crop(b)).strip() for l, b in boxes.items()}, "pytesseract OCR on layout regions"
    except Exception:
        return {l: str(row[dc]) for l, dc, _ in FIELDS}, "stored text per layout region (simulated OCR; install Tesseract + pytesseract for real OCR)"


# ---------------- NER and case-note classification ----------------
ENT = {"CUSTOMER": r"\bC\d{5}\b", "DEVICE": r"\bD[RN]?\d+\b", "BENEFICIARY": r"\bB[RSN]?\d+(?:_\d+)?\b",
       "AMOUNT": r"(?:INR|Rs\.?)\s?[\d,]+(?:\.\d+)?", "DATE": r"\b\d{4}-\d{2}-\d{2}\b", "ID_NUMBER": r"\bID\d{7,8}\b"}


def extract_entities(text):
    rows = [(k, m.group(0)) for k, p in ENT.items() for m in re.finditer(p, text)]
    return pd.DataFrame(rows, columns=["entity_type", "text"]).drop_duplicates()


_TPL = {
    "mule": ["Credits from several unrelated payers were forwarded within hours to {ben}; device {dev} is also used by other accounts.",
             "Pass-through of INR {amt} in one day to beneficiary {ben}, shared device {dev} across linked accounts."],
    "ato": ["Login from new device {dev} at night, contact details changed, then {n} transfers to new payee {ben}.",
            "Customer reports unfamiliar sign-in; INR {amt} moved to newly added payee {ben} from device {dev}."],
    "scam": ["Customer was contacted by a stranger and voluntarily sent INR {amt} to new payee {ben}.",
             "Victim made a large urgent transfer of INR {amt} to {ben} after a phone call from a fraudster."],
    "structuring": ["Repeated deposits of INR 9,900 just below the reporting threshold, {n} times in a week.",
                    "Cash split into {n} amounts under the limit, around INR 9,950 each, to avoid reporting."],
    "benign": ["Salary run caused {n} same-day payments to known merchants; closed as legitimate.",
               "Festival shopping burst of {n} payments to {ben}; activity consistent with history, no concern."]}
_NOISE = ["The customer was contacted by phone.", "A new device was noted.", "Several payments were reviewed.", "Funds moved quickly."]


def _notes(n=500, seed=C.SEED):
    rng = np.random.default_rng(seed); X, y = [], []
    for _ in range(n):
        lab = rng.choice(list(_TPL)); t = rng.choice(_TPL[lab])
        s = t.format(ben=f"B{rng.integers(99)}_{rng.integers(5)}", dev=f"D{rng.integers(999)}", amt=f"{int(rng.integers(5, 90)) * 1000:,}", n=int(rng.integers(3, 14)))
        if rng.random() < .35: s += " " + str(rng.choice(_NOISE))
        X.append(s); y.append(lab)
    return X, y


def train_note_classifier():
    X, y = _notes(); a, b, ya, yb = train_test_split(X, y, test_size=.3, random_state=C.SEED, stratify=y)
    m = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=2000)).fit(a, ya)
    p = m.predict(b)
    return m, dict(accuracy=round(accuracy_score(yb, p), 3), macro_f1=round(f1_score(yb, p, average="macro"), 3), n_test=len(b))
