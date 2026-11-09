"""LLM investigator copilot: retrieval-grounded, cited, uncertainty-aware, refuses unsupported / high-impact requests."""
import os, re, time
import numpy as np
import pandas as pd
import config as C
from src import responsible_ai as rai
from src.detection import RULES, reason_text
from src.retrieval import SUPPORT

QUERY = {"mule": "money mule accounts receive many credits forward funds quickly shared beneficiaries share device address structuring",
         "ato": "account takeover new device unusual hour burst high-value transfers new payees login",
         "scam": "authorised push payment scam victim large transfer new payee fraudster",
         "structuring": "structuring repeated amounts just below reporting threshold review",
         "identity": "identity document mismatch name date of birth enhanced verification secondary document",
         "general": "velocity review many transactions escalation policy human approval"}
HIGH_IMPACT = re.compile(r"\b(freeze|block|reject|close the account|file (a |the )?(sar|report)|tip off|tell the customer)\b", re.I)


def typology(f):
    if f.new_dev_share > .05 and f.max_to_income >= .3: return "ato"
    if f.pass_through > .25 or (f.g_component >= 3 and f.near_thr_cnt >= 1): return "mule"
    if f.near_thr_cnt >= 3: return "structuring"
    if f.max_to_income >= 2: return "scam"
    if f.doc_mismatch > .08: return "identity"
    return "general"


def _first_sentence(t): return re.split(r"(?<=[.!?])\s", t.strip())[0]


class Copilot:
    def __init__(self, X):
        self.X = X; self.R = X["retriever"]; self.F = X["F"]; self.S = X["det"]["S"]; self.tier = X["det"]["tier"]
        self.rh = X["det"]["rule_hits"]; self.P = X["P"]; self.reasoner = X["det"]["reasoner_full"]; self.cust = X["cust"].set_index("customer_id")
        self.cases = self.R.chunks.set_index("chunk_id").case_type.to_dict()

    # ---- main workflow ------------------------------------------------------------------------------------------
    def investigate(self, cid, use_llm=False, mode="hybrid_rerank", multimodal=True):
        t0 = time.perf_counter(); f = self.F.loc[cid]; s = float(self.S.loc[cid, "ens_full"]); tier = self.tier[cid]
        reasons = [reason_text(a, v) for a, _, v in self.reasoner.explain(self.F, cid, 4)]
        hits = [n for n, h in zip([r[0] for r in RULES], self.rh.loc[cid].values) if h]
        neigh = [(o, self.P[cid][o]["res"]) for o in self.P.neighbors(cid)] if cid in self.P else []
        typ = typology(f); q = QUERY[typ] + (" high-risk jurisdiction enhanced due diligence" if f.xborder_share > .05 else "")
        res = self.R.search(q, mode, 6, multimodal); safe, quar = rai.sanitize_results(res); safe = safe[:3]
        conf = safe[0]["score"] if safe else 0.0; decline = conf < C.RETRIEVAL_CONFIDENCE_MIN or not safe
        data_sents = [dict(kind="claim", cites=["D1"], text=f"Customer {cid} has an ensemble risk score of {s:.2f} (tier {tier}); main drivers: " + ("; ".join(reasons[:3]) or "no dominant driver") + "."),
                      dict(kind="claim", cites=["D2"], text="Rule scenarios triggered: " + (", ".join(hits) or "none") + "."),
                      dict(kind="claim", cites=["D3"], text=f"{len(neigh)} other customer(s) are directly linked through shared devices, addresses or beneficiaries; the connected network has {int(f.g_component)} customer(s).")]
        ev_sents = []
        if not decline:
            for i, e in enumerate(safe, 1):
                body = _first_sentence(e["text"]) if e["text"] else "Visual evidence shows " + e["visual_desc"] + "."
                ev_sents.append(dict(kind="claim", cites=[f"P{i}"], text=f"{e['doc_type'].title()} {e['chunk_id']} (p.{e['page']}) states: {body}"))
        meta = [dict(kind="meta", cites=[], text=("Retrieved evidence is not strong enough for a definitive conclusion; a manual evidence search is requested." if decline else f"The behaviour is consistent with the '{typ}' pattern, which is a hypothesis for the investigator to confirm.")),
                dict(kind="meta", cites=[], text=f"Uncertainty: retrieval confidence {conf:.2f}; the score is a calibrated probability estimated on synthetic data; this summary is advisory and any customer-impacting action needs authorised human approval.")]
        narrative = data_sents + ev_sents + meta; llm_used = False
        if use_llm and not decline:
            alt = self._llm(cid, data_sents, safe)
            if alt: narrative, llm_used = alt + meta, True
        cites = {"D1": "Feature store + ensemble score", "D2": "Rule engine output", "D3": "Entity graph (shared device/address/beneficiary)"}
        cites.update({f"P{i}": f"{e['chunk_id']} p.{e['page']}" for i, e in enumerate(safe, 1)})
        g = self.check_grounded(narrative, safe)
        return dict(cid=cid, score=s, tier=tier, reasons=reasons, hits=hits, neighbours=neigh, typology=typ, query=q, evidence=safe, quarantined=quar,
                    confidence=conf, declined=decline, narrative=narrative, citations=cites, grounded=g, llm_used=llm_used, latency=time.perf_counter() - t0)

    # ---- grounding / citation checks -------------------------------------------------------------------------------
    def check_grounded(self, narrative, evidence):
        ev = {f"P{i}": set(re.findall(r"\w+", (e["title"] + " " + e["text"] + " " + e["visual_desc"]).lower())) for i, e in enumerate(evidence, 1)}
        claims = [n for n in narrative if n["kind"] == "claim"]; bad = []
        for n in claims:
            ok = bool(n["cites"])
            for c in n["cites"]:
                if c.startswith("P"):
                    toks = set(re.findall(r"\w+", n["text"].lower()))
                    ok = ok and c in ev and len(toks & ev[c]) / max(len(toks), 1) >= .6
            if not ok: bad.append(n["text"])
        cited = [c["cites"] for c in claims]; flat = [x for c in cited for x in c if x.startswith("P")]
        return dict(claims=len(claims), unsupported=len(bad), unsupported_rate=len(bad) / max(len(claims), 1), unsupported_text=bad, cited_P=len(flat))

    def citation_precision(self, res):
        sup = set(SUPPORT[res["typology"]]) | {c for c, t in self.cases.items() if t == res["typology"]}
        ids = [e["chunk_id"] for e in res["evidence"]]
        return (sum(i in sup for i in ids) / len(ids)) if ids else np.nan

    def _llm(self, cid, data_sents, evidence):
        if not os.getenv("ANTHROPIC_API_KEY"): return None
        try:
            import anthropic
            ctx = "\n".join(f"[D{i + 1}] {d['text']}" for i, d in enumerate(data_sents)) + "\n" + "\n".join(f"[P{i}] {e['chunk_id']}: {e['text'] or e['visual_desc']}" for i, e in enumerate(evidence, 1))
            msg = anthropic.Anthropic().messages.create(model=C.LLM_MODEL, max_tokens=500, messages=[{"role": "user", "content":
                "You assist a financial-crime investigator. Using ONLY the evidence below, write 4-6 short sentences. End every sentence with its citation tags like [D1] or [P2]. "
                "Do not decide or recommend freezing/rejecting/reporting. If evidence is insufficient say so.\n\n" + rai.mask_pii(ctx)}])
            out = []
            for sent in re.split(r"(?<=[.!?])\s+", msg.content[0].text.strip()):
                tags = re.findall(r"\[((?:D|P)\d+)\]", sent); out.append(dict(kind="claim", cites=tags, text=re.sub(r"\s*\[(?:D|P)\d+\]", "", sent)))
            return out
        except Exception: return None

    # ---- free-form Q&A with refusal ------------------------------------------------------------------------------------
    def ask(self, q):
        if HIGH_IMPACT.search(q):
            e = self.R.search("human approval account freezing AI advisory", "hybrid_rerank", 1)[0]
            return dict(refused=True, answer="I cannot take or recommend that customer-impacting action. It requires an authorised human decision.", citations=[f"{e['chunk_id']} p.{e['page']}"])
        res, _ = rai.sanitize_results(self.R.search(rai.mask_pii(q), "hybrid_rerank", 3))
        if not res or res[0]["score"] < C.RETRIEVAL_CONFIDENCE_MIN: return dict(refused=True, answer="Retrieval confidence is too low to answer; please run a manual evidence search.", citations=[])
        return dict(refused=False, answer=" ".join(_first_sentence(r["text"]) if r["text"] else r["visual_desc"] for r in res[:2]), citations=[f"{r['chunk_id']} p.{r['page']}" for r in res[:2]])
