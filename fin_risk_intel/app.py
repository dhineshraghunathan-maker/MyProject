"""Streamlit demo UI: Governed AI/ML financial risk-intelligence prototype (BITS WILP dissertation S1-26_AIMLCZG628T)."""
import json
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import config as C
from src import nlp_docai as nd, responsible_ai as rai
from src.detection import RULES, FEAT_ONB, reason_text, topk_idx
from src.pipeline import get_state

st.set_page_config(page_title="Governed Risk-Intelligence Prototype", layout="wide", page_icon="🛡️")


@st.cache_resource(show_spinner="Refreshing synthetic data and models with the latest schema (about 15 s)...")
def load(): return get_state(force=True)


X = load(); F, S, det, cust = X["F"], X["det"]["S"], X["det"], X["cust"].set_index("customer_id")
tier, R, cop, P, te = det["tier"], X["retriever"], X["copilot"], X["P"], det["te"]
te_ids = F.index[te]; queue = S.loc[te_ids].sort_values("ens_full", ascending=False).index
BADGE = {"ESCALATE": ":red[**ESCALATE**]", "REVIEW": ":orange[**REVIEW**]", "LOW": ":green[**LOW**]"}
fmt = lambda c: f"{c} · {tier[c]} · score {S.ens_full[c]:.2f}"

st.title("🛡️ AI & ML Powered Strategic Financial Systems")
st.caption("Governed risk-intelligence prototype · BITS Pilani WILP · S1-26_AIMLCZG628T · Dhinesh Raghunathan (2024AD05461) · "
           "all data is synthetic · the LLM copilot never makes a customer-impacting decision")
tabs = st.tabs(["1 Overview", "2 Account opening", "3 Surveillance & fraud", "4 Graph intelligence", "5 Investigator copilot",
                "6 Retrieval lab", "7 Model evaluation", "8 Responsible AI", "9 Hypotheses"])

# ------------------------------------------------------------------ 1 Overview
with tabs[0]:
    c = st.columns(5); y = det["y"]
    c[0].metric("Customers", f"{len(F):,}"); c[1].metric("Transactions", f"{len(X['tx']):,}"); c[2].metric("Fraud rate", f"{y.mean():.1%}")
    c[3].metric("Test customers / review capacity", f"{te.sum():,} / {det['K']}"); c[4].metric("Alerts (REVIEW+ESCALATE)", int((tier[te_ids] != 'LOW').sum()))
    st.subheader("Architecture (Figure 1 of the outline: components C1-C7, links L1-L8)")
    st.graphviz_chart('''digraph G { rankdir=LR; node [shape=box, style="rounded,filled", fontname=Helvetica, fontsize=11];
 SRC [label="Source Systems\\nKYC, payments, docs", fillcolor="#dbeafe"]; DF [label="Data Foundation\\nquality, entities, features", fillcolor="#dbeafe"];
 ML [label="AI / ML Engine\\nrules, ML, graph, NLP", fillcolor="#dbeafe"]; DEC [label="Decisioning\\nensemble + human approval", fillcolor="#dbeafe"];
 IR [label="Multimodal RAG / IR\\nhybrid search, rerank, citations", fillcolor="#fde68a"]; LLM [label="LLM Copilot\\ngrounded summary + uncertainty", fillcolor="#fde68a"];
 GOV [label="Governance, Security, Responsible AI, MLOps", fillcolor="#e5e7eb", style="filled,dashed"];
 SRC->DF [label="L1"]; DF->ML [label="L2"]; ML->DEC [label="L3"]; DF->IR [label="L4"]; ML->IR [label="L5"]; IR->LLM [label="L6"]; LLM->DEC [label="L7"];
 GOV->DF [style=dashed, label="L8"]; GOV->ML [style=dashed]; GOV->IR [style=dashed]; GOV->LLM [style=dashed]; GOV->DEC [style=dashed]}''')
    st.subheader("Objective → implementation → where to see it")
    st.dataframe(pd.DataFrame([
        ("Canonical data model", "src/data_gen.py, data/*.csv, data_dictionary.csv", "Overview, files in data/"),
        ("Explainable onboarding risk, calibrated", "src/nlp_docai.py, src/detection.py (Onboarding LR)", "2 Account opening, 7 Evaluation"),
        ("Rule / supervised / anomaly / graph baselines", "src/detection.py, src/features.py", "3, 4, 7"),
        ("NLP + document AI (NER, OCR/layout, classification)", "src/nlp_docai.py", "2 Account opening"),
        ("BM25 / dense / hybrid + rerank + filters", "src/retrieval.py", "6 Retrieval lab"),
        ("Multimodal evidence with page citations", "src/retrieval.py (image/table chunks)", "5 Copilot, 6 Retrieval lab"),
        ("LLM copilot: cited, refuses unsupported", "src/copilot.py", "5 Investigator copilot"),
        ("Responsible AI controls (fairness, privacy, safety, oversight)", "src/responsible_ai.py", "8 Responsible AI")],
        columns=["Objective (3.2)", "Implementation", "Demo tab"]), hide_index=True)
    with st.expander("Suggested 5-minute demo script for the professor"):
        st.markdown("1. **Overview** – architecture and traceability.  2. **Account opening** – pick a customer with a red-boxed document field; show reconciliation, CDD score and reason codes.  "
                    "3. **Graph intelligence** – open a top ESCALATE customer, show the mule network.  4. **Investigator copilot** – run the investigation, show cited evidence incl. a scanned page, "
                    "the uncertainty line, then try *'Freeze this account'* (refusal) and submit a human decision (audit log).  5. **Retrieval lab** – compare BM25 / dense / hybrid and text-only vs multimodal.  "
                    "6. **Model evaluation** – move the review-capacity slider.  7. **Responsible AI + Hypotheses** – fairness, red-team results and honest H1–H6 verdicts.")

# ------------------------------------------------------------------ 2 Account opening
with tabs[1]:
    st.subheader("Account-opening intelligence: identity & document reconciliation, explainable CDD risk")
    pool = list(S.loc[te_ids].sort_values("onboarding_p", ascending=False).index[:150]) + list(queue[:5])
    cid = st.selectbox("Applicant", pool, format_func=lambda c: f"{c} · CDD risk {S.onboarding_p[c]:.2f} · doc mismatch {F.doc_mismatch[c]:.2f}", key="kyc_sel")
    row = X["kyc"].loc[cid]; path, boxes = nd.render_kyc_image(cid, row); ext, how = nd.layout_extract(path, boxes, row)
    a, b = st.columns([1.1, 1])
    a.image(path, caption="Synthetic identity document (red box = differs from application)")
    t = pd.DataFrame([
        ("Name", row.declared_name, ext["Name"]),
        ("Date of birth", row.declared_dob, ext["Date of birth"]),
        ("Address", row.declared_address, ext["Address"]),
        ("Occupation / Profession", row.declared_occupation, ext["Occupation / Profession"]),
        ("Source of income", row.declared_source_of_income, ext["Source of income"]),
        ("Income range", row.declared_income_range, ext["Income range"]),
        ("PEP", row.declared_pep, ext["PEP"]),
        ("ID type", row.declared_id_type, ext["ID type"]),
        ("ID number", row.declared_id_number, ext["ID number"]),
    ], columns=["Field", "Declared on application", "Extracted from document"])
    t["Match"] = np.where(t.iloc[:, 1].str.lower() == t.iloc[:, 2].str.lower(), "✅", "⚠️"); b.dataframe(t, hide_index=True); b.caption("Extraction: " + how)
    pep_choice = st.selectbox("Politically Exposed Person (PEP)", ["No", "Yes"], index=(1 if row.declared_pep == "Yes" else 0), key=f"pep_{cid}")
    b.metric("Document mismatch score", f"{row.doc_mismatch:.2f}", help="1 − weighted similarity of name, DOB and address")
    b.metric("Calibrated CDD risk (onboarding model)", f"{S.onboarding_p[cid]:.1%}")
    st.markdown("**Reason codes (onboarding model):** " + ("; ".join(reason_text(f, v) for f, _, v in det["reasoner_onb"].explain(F, cid, 3)) or "none above baseline"))
    st.divider(); st.subheader("NLP: entity extraction and case-note classification")
    note = st.text_area("Case note", f"Login from new device DN{cid[1:]} at night; INR 45,000 moved to new payee BN{cid[1:]}_1 on 2026-03-14 for customer {cid}.")
    e = nd.extract_entities(note); st.dataframe(e, hide_index=True)
    p = X["notes_clf"].predict_proba([note])[0]; st.write("Predicted typology: **%s** (%.0f%%)" % (X["notes_clf"].classes_[p.argmax()], 100 * p.max()), "· classifier hold-out", X["notes_m"])

# ------------------------------------------------------------------ 3 Surveillance
with tabs[2]:
    st.subheader("Transaction surveillance, monitoring and fraud alert queue")
    sc = X["scen"]; raw = sc.groupby("scenario").size().rename("raw alert-days"); dd = sc.drop_duplicates(["customer_id", "scenario"]).groupby("scenario").size().rename("after de-duplication")
    st.dataframe(pd.concat([raw, dd], axis=1).assign(reduction=lambda d: (1 - d.iloc[:, 1] / d.iloc[:, 0]).map("{:.0%}".format)))
    mx = int(sc.day.max()); d0 = int(sc.day.min()); day = st.slider("Stream replay: alerts up to day", d0, mx, mx, help="Simulates streaming by replaying the test period")
    up = sc[sc.day <= day]; st.plotly_chart(px.histogram(up, x="day", color="scenario", nbins=40, title=f"Scenario alerts up to day {day}"))
    st.markdown("**Prioritised customer alert queue** (ensemble score → tier; fixed review capacity = %d)" % det["K"])
    qd = pd.DataFrame({"customer": queue[:60], "tier": tier[queue[:60]].values, "ensemble": S.ens_full[queue[:60]].round(3).values, "ML+graph p": S.mlg_p[queue[:60]].round(3).values,
                       "rule hits": [int(det["rule_hits"].loc[c].sum()) for c in queue[:60]], "anomaly pct": S.anomaly[queue[:60]].round(2).values,
                       "top reason": [reason_text(*(lambda r: (r[0][0], r[0][2]))(det["reasoner_full"].explain(F, c, 1))) if det["reasoner_full"].explain(F, c, 1) else "-" for c in queue[:60]]})
    st.dataframe(qd, hide_index=True)

# ------------------------------------------------------------------ 4 Graph
with tabs[3]:
    st.subheader("Entity & graph intelligence: connected risk via shared devices, addresses, beneficiaries")
    g1 = st.selectbox("Focus customer", list(queue[:60]), format_func=fmt, key="g_sel"); reveal = st.checkbox("Reveal investigated outcome (ground truth)", False)
    G = nx.ego_graph(P, g1, radius=2)
    if G.number_of_nodes() > 45: G = G.subgraph(sorted(G.nodes, key=lambda n: nx.shortest_path_length(G, g1, n))[:45])
    pos = nx.spring_layout(G, seed=C.SEED); ex, ey = [], []
    for u, v in G.edges: ex += [pos[u][0], pos[v][0], None]; ey += [pos[u][1], pos[v][1], None]
    col = {"ESCALATE": "#dc2626", "REVIEW": "#f59e0b", "LOW": "#10b981"}
    fig = go.Figure(go.Scatter(x=ex, y=ey, mode="lines", line=dict(width=1, color="#9ca3af"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=[pos[n][0] for n in G], y=[pos[n][1] for n in G], mode="markers+text", text=[n[1:] if n != g1 else "★" + n[1:] for n in G], textposition="top center",
                             marker=dict(size=[26 if n == g1 else 16 for n in G], color=[col[tier[n]] for n in G], line=dict(width=[3 if (reveal and cust.is_fraud[n]) else 0 for n in G], color="black")),
                             hovertext=[f"{n} · {cust.segment[n]}/{cust.region[n]} · score {S.ens_full[n]:.2f} · {tier[n]}" + (f" · outcome {cust.fraud_type[n]}" if reveal else "") for n in G], hoverinfo="text"))
    fig.update_layout(showlegend=False, height=520, xaxis_visible=False, yaxis_visible=False, margin=dict(l=0, r=0, t=10, b=0)); st.plotly_chart(fig)
    st.caption("Colour = tier (red ESCALATE, amber REVIEW, green LOW)" + ("; black outline = confirmed fraud outcome" if reveal else ""))
    lk = pd.DataFrame([(o, ", ".join(P[g1][o]["res"]), tier[o]) for o in P.neighbors(g1)], columns=["linked customer", "shared resources", "tier"]); st.dataframe(lk, hide_index=True)
    st.write(f"Graph features: degree {int(F.g_degree[g1])}, component size {int(F.g_component[g1])}, community size {int(F.g_community[g1])}, pagerank {F.g_pagerank[g1]:.4f}")

# ------------------------------------------------------------------ 5 Copilot
with tabs[4]:
    st.subheader("Investigator view: score, reason codes, linked entities, cited evidence, drafted narrative with uncertainty")
    c1, c2, c3 = st.columns([2, 1, 1]); sel = c1.selectbox("Alert", list(queue[:60]), format_func=fmt, key="cp_sel")
    mm = c2.checkbox("Multimodal evidence", True); llm = c3.checkbox("Use LLM (needs ANTHROPIC_API_KEY)", False)
    if st.button("Run investigation", type="primary") or st.session_state.get("cp_last") == sel:
        st.session_state["cp_last"] = sel; r = cop.investigate(sel, use_llm=llm, multimodal=mm)
        k = st.columns(4); k[0].metric("Risk score", f"{r['score']:.2f}"); k[1].markdown("**Tier**<br>" + BADGE[r["tier"]], unsafe_allow_html=True)
        k[2].metric("Retrieval confidence", f"{r['confidence']:.2f}"); k[3].metric("Typology hypothesis", r["typology"])
        L, Rr = st.columns(2)
        with L:
            st.markdown("**Reason codes**"); [st.markdown(f"- {x}") for x in r["reasons"]]
            st.markdown("**Rules triggered**: " + (", ".join(r["hits"]) or "none")); st.markdown(f"**Linked entities**: {len(r['neighbours'])} customers")
            if r["neighbours"]: st.dataframe(pd.DataFrame([(o, ", ".join(rs)) for o, rs in r["neighbours"][:8]], columns=["customer", "shared"]), hide_index=True)
        with Rr:
            st.markdown("**Drafted narrative (advisory, cited)**")
            for n in r["narrative"]:
                st.markdown(f"- {n['text']} " + (f"`{' '.join('[' + c + ']' for c in n['cites'])}`" if n["cites"] else "_(uncertainty/procedure)_"))
            g = r["grounded"]; st.caption(f"Groundedness: {g['claims'] - g['unsupported']}/{g['claims']} claims supported · unsupported rate {g['unsupported_rate']:.0%}" + (" · LLM text" if r["llm_used"] else " · extractive draft"))
            if r["declined"]: st.warning("Low retrieval confidence: definitive conclusion declined; manual evidence search requested.")
        if r["quarantined"]: st.error(f"🚫 {len(r['quarantined'])} retrieved chunk(s) quarantined by the prompt-injection guard: " + ", ".join(q["chunk_id"] for q in r["quarantined"]))
        st.markdown("**Retrieved evidence with citations**")
        for i, e in enumerate(r["evidence"], 1):
            with st.container(border=True):
                st.markdown(f"**[P{i}] {e['chunk_id']}** · {e['doc_type']} · page {e['page']} · score {e['score']:.2f}")
                if e["modality"] == "image": st.image(R.page_image(e["chunk_id"]), width=420); st.caption(e["visual_desc"])
                else: st.write(e["text"])
        with st.form("decision"):
            st.markdown("**Human decision (required for any customer-impacting outcome)**")
            dc = st.radio("Reviewer decision", ["Agree: proceed with monitoring (LOW)", "Agree: keep in REVIEW", "Escalate to senior review", "Request manual evidence search"], horizontal=True)
            who = st.text_input("Reviewer", "Investigator-1"); note = st.text_area("Notes"); rate = st.slider("Narrative quality rating", 1, 5, 4); flag = st.checkbox("I noticed an unsupported statement")
            if st.form_submit_button("Submit & write audit record"):
                rai.log_decision(customer=sel, tier=r["tier"], score=round(r["score"], 3), decision=dc, reviewer=who, notes=note, rating=rate, unsupported_noticed=flag, evidence=[e["chunk_id"] for e in r["evidence"]], llm_used=r["llm_used"])
                st.success("Decision recorded in artifacts/audit_log.jsonl")
    st.divider(); q = st.text_input("Ask the copilot a question (try: 'Freeze this account now' or 'what is the rule for structuring?')")
    if q:
        a = cop.ask(q); (st.warning if a["refused"] else st.info)(a["answer"] + ("  — " + "; ".join(a["citations"]) if a["citations"] else ""))
    aud = rai.read_audit()
    if not aud.empty: st.markdown("**Audit trail**"); st.dataframe(aud.tail(10), hide_index=True)

# ------------------------------------------------------------------ 6 Retrieval lab
with tabs[5]:
    st.subheader("Information retrieval lab: BM25 vs dense vs hybrid (RRF) vs rerank; text-only vs multimodal; version filter")
    q = st.text_input("Query", "scanned identity page red rejection stamp"); o1, o2 = st.columns(2)
    mmx = o1.checkbox("Multimodal index", True, key="lab_mm"); flt = o2.checkbox("Only current, authorised versions", True)
    cols = st.columns(4)
    for col, m in zip(cols, R.MODES):
        col.markdown(f"**{m}**")
        for h in R.search(q, m, 4, mmx, flt): col.caption(f"{h['rank']}. `{h['chunk_id']}` ({h['score']:.2f})")
    ev = X["rev"]; sub = st.radio("Benchmark subset", ["all", "exact", "paraphrase", "image"], horizontal=True)
    t = ev[ev.subset == sub].drop(columns="subset").set_index("config"); st.dataframe(t.round(3)); st.plotly_chart(px.bar(t.reset_index(), x="config", y=["recall_at_5", "MRR", "nDCG_at_5"], barmode="group"))
    st.caption(f"Benchmark: {int(ev[ev.subset=='all'].n.iloc[0])} labelled queries (exact-term, paraphrase and image-dependent). Dense = TF-IDF+LSA offline; set USE_TRANSFORMERS=1 for MiniLM + cross-encoder.")

# ------------------------------------------------------------------ 7 Evaluation
with tabs[6]:
    st.subheader("Detection evaluation (time-based split; test = later account openings)")
    frac = st.slider("Review capacity (share of test customers investigators can review)", .01, .15, C.REVIEW_CAPACITY, .01); K = int(np.ceil(frac * te.sum())); yt = det["y"][te]
    names = {"rule_score": "Rule-only", "anomaly": "Anomaly", "ml_p": "ML", "ens_rules_ml": "Rules+ML", "mlg_p": "ML+graph", "ens_full": "Full ensemble"}
    rows = [dict(model=n, K=K, precision_at_K=yt[topk_idx(S[c].values[te], K)].mean(), recall_at_K=yt[topk_idx(S[c].values[te], K)].sum() / yt.sum()) for c, n in names.items()]
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True); st.plotly_chart(px.bar(pd.DataFrame(rows), x="model", y=["precision_at_K", "recall_at_K"], barmode="group"))
    st.markdown("**Ablations (at default capacity)**"); st.dataframe(det["metrics"].drop(columns="score").round(3), hide_index=True)
    a, b = st.columns(2)
    f1 = go.Figure([go.Scatter(x=v[0], y=v[1], name=k, mode="lines") for k, v in det["pr"].items()]).update_layout(title="Precision-recall (test)", xaxis_title="recall", yaxis_title="precision"); a.plotly_chart(f1)
    f2 = go.Figure([go.Scatter(x=v[0], y=v[1], name=k, mode="lines+markers") for k, v in det["calibration"].items()] + [go.Scatter(x=[0, 1], y=[0, 1], name="perfect", line=dict(dash="dash"))]).update_layout(title="Reliability curves", xaxis_title="predicted", yaxis_title="observed"); b.plotly_chart(f2)
    st.plotly_chart(px.bar(det["importance"].head(10)[::-1], orientation="h", title="Permutation importance (ΔPR-AUC, ML+graph)"))
    ce = X["copilot_eval"]; st.markdown("**Copilot evaluation on all %d alerts**" % len(ce)); st.dataframe(pd.DataFrame({"unsupported statement rate": [ce.unsupported_rate.mean()], "citation precision": [ce.citation_precision.mean()], "decline rate": [ce.declined.mean()],
                 "typology agreement (true fraud)": [ce.typology_match.mean()], "mean latency (s)": [ce.latency_s.mean()]}).round(3), hide_index=True)

# ------------------------------------------------------------------ 8 Responsible AI
with tabs[7]:
    st.subheader("Responsible AI controls measured as outcomes")
    attr = st.selectbox("Segment attribute", list(X["fair"])); t, gp = X["fair"][attr]
    st.dataframe(t.round(3), hide_index=True); ok = max(gp["FPR_gap"], gp["FNR_gap"]) <= C.FAIRNESS_GAP_THRESHOLD
    st.metric("FPR gap / FNR gap", f"{gp['FPR_gap']:.3f} / {gp['FNR_gap']:.3f}", delta="within threshold" if ok else f"exceeds {C.FAIRNESS_GAP_THRESHOLD:.2f}", delta_color="normal" if ok else "inverse")
    st.caption(f"Smallest group has {gp['min_positives']} true positives, so FNR gaps are statistically noisy. Age, region and segment are not model inputs.")
    sf = X["safety"]; c = st.columns(4)
    c[0].metric("Prompt-injection block rate", f"{sf['injection']['block_rate']:.0%}", f"{sf['injection']['false_positives']} false positives"); c[1].metric("PII leak rate after masking", f"{sf['privacy']['leak_rate']:.0%}")
    c[2].metric("High-impact action refusal", f"{sf['refusal']['rate']:.0%}"); c[3].metric("Poisoned chunks quarantined in alerts", sf["quarantined_in_alerts"])
    with st.expander("Privacy masking examples"): st.dataframe(pd.DataFrame(sf["privacy"]["examples"], columns=["input", "masked"]), hide_index=True)
    st.markdown("**Drift monitoring (PSI train vs test; >0.25 = significant)**"); st.dataframe(det["psi"].sort_values("PSI_train_vs_test", ascending=False).head(8).round(3), hide_index=True)
    import pathlib; cards = sorted((C.ART / "model_cards").glob("*.md")); pick = st.selectbox("Model card", cards, format_func=lambda p: p.stem); st.markdown(pick.read_text(encoding="utf-8"))
    st.markdown("**Retrieval-corpus inventory**"); st.dataframe(X["inv"], hide_index=True)
    aud = rai.read_audit(); st.markdown("**Human decisions and ratings (accountability)**"); st.dataframe(aud if not aud.empty else pd.DataFrame({"info": ["No decisions logged yet - submit one in tab 5"]}), hide_index=True)

# ------------------------------------------------------------------ 9 Hypotheses
with tabs[8]:
    st.subheader("Hypotheses H1-H6: measured results (testable hypotheses, not established findings)")
    h = X["hyp"]; st.dataframe(h[["id", "test", "result", "verdict", "note"]], hide_index=True)
    st.info("Honest reading: results come from SYNTHETIC data. H3 depends on the dense/rerank models used; H5 is a time-and-motion simulation; H6 fairness gaps are noisy with few positives per group.")
    m, a_ = X["handling"]; st.write(f"Simulated handling time per alert: manual {m:.1f} min vs assisted {a_:.1f} min (assumptions in config.HANDLING).")
    st.download_button("Download hypothesis results (JSON)", json.dumps(h.to_dict("records"), default=str, indent=2), "hypothesis_results.json")
