"""Rules, calibrated supervised models, anomaly models, graph features, ensemble, explainability."""
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import config as C

FEAT_ONB = ["occ_risk", "country_risk", "doc_mismatch", "log_income"]   # no age/region/segment (fairness)
FEAT_BEH = ["n_txn", "total_amt", "mean_amt", "std_amt", "max_amt", "near_thr_cnt", "night_share", "n_ben", "new_dev_share",
            "xborder_share", "max_daily_cnt", "pass_through", "max_to_income", "peer_dev"]
FEAT_GRAPH = ["g_degree", "g_wdegree", "g_component", "g_pagerank", "g_community"]

RULES = [("R1 Structuring (>=3 txns of 9,000-9,999)", lambda f: f.near_thr_cnt >= 3, 1.0),
         ("R2 High-value vs income (>=3x monthly)", lambda f: f.max_to_income >= 3, 1.0),
         ("R3 Night activity (>8% of txns 00-06h)", lambda f: f.night_share > .08, .8),
         ("R4 Velocity (>=10 txns in a day)", lambda f: f.max_daily_cnt >= 10, 1.0),
         ("R5 Document mismatch (>0.08)", lambda f: f.doc_mismatch > .08, 1.0),
         ("R6 Non-home device (>5% of txns)", lambda f: f.new_dev_share > .05, 1.0),
         ("R7 High-risk country (>5% of txns)", lambda f: f.xborder_share > .05, .6)]

TXT = {"near_thr_cnt": "{v:.0f} transactions just below the 10,000 threshold", "night_share": "{v:.0%} of activity between 00:00-06:00",
       "new_dev_share": "{v:.0%} of transactions from a non-home device", "max_daily_cnt": "up to {v:.0f} transactions in one day",
       "pass_through": "{v:.0%} of credited value forwarded the same day", "max_to_income": "largest transfer is {v:.1f}x monthly income",
       "xborder_share": "{v:.0%} of transactions linked to a high-risk country", "doc_mismatch": "identity document differs from application (score {v:.2f})",
       "g_degree": "{v:.0f} other customers share a device, address or beneficiary", "g_wdegree": "{v:.0f} shared-resource links to other customers",
       "g_component": "part of a linked network of {v:.0f} customers", "g_community": "community of {v:.0f} closely linked customers",
       "occ_risk": "higher-risk declared occupation ({v:.1f})", "country_risk": "linked to a high-risk jurisdiction", "peer_dev": "average amount {v:.1f} sd above segment peers",
       "n_ben": "{v:.0f} distinct beneficiaries", "max_amt": "largest single amount {v:,.0f}", "g_pagerank": "central position in the entity graph",
       "mean_amt": "high average amount {v:,.0f}", "total_amt": "high total value {v:,.0f}", "std_amt": "volatile amounts", "n_txn": "{v:.0f} transactions", "log_income": "low declared income"}


def reason_text(feat, v): return TXT.get(feat, feat + " = {v:.3g}").format(v=v)


class Reasoner:
    """Local explanation = coefficient x standardised value of an interpretable surrogate (reason codes)."""
    def __init__(self, F, y, tr, cols):
        self.cols = cols; X = self._x(F[cols]); self.sc = StandardScaler().fit(X[tr])
        self.lr = LogisticRegression(C=.5, class_weight="balanced", max_iter=3000).fit(self.sc.transform(X[tr]), y[tr])

    def _x(self, df): return np.log1p(df.clip(lower=0).values)

    def explain(self, F, cid, n=4):
        z = self.sc.transform(self._x(F.loc[[cid], self.cols])); c = z[0] * self.lr.coef_[0]
        return [(self.cols[i], float(c[i]), float(F.loc[cid, self.cols[i]])) for i in np.argsort(-c)[:n] if c[i] > 0]


def topk_idx(s, K, seed=C.SEED):
    return np.argsort(-(np.asarray(s) + 1e-9 * np.random.default_rng(seed).random(len(s))))[:K]


def metrics_row(y, s, K, prob=True):
    o = topk_idx(s, K)
    return dict(pr_auc=average_precision_score(y, s), precision_at_K=y[o].mean(), recall_at_K=y[o].sum() / max(y.sum(), 1),
                brier=brier_score_loss(y, s) if prob else np.nan)


def boot_diff(y, a, b, K, n=300, seed=C.SEED):
    rng = np.random.default_rng(seed); d = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y)); yy = y[i]
        d.append(yy[topk_idx(a[i], K)].mean() - yy[topk_idx(b[i], K)].mean())
    return float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def run_detection(F, cust):
    ci = cust.set_index("customer_id").loc[F.index]; y = ci.is_fraud.values.astype(int); od = ci.open_day.values
    tr = od <= np.quantile(od, .65); te = ~tr                                 # time-based split on account-opening date
    rh = pd.DataFrame({n: fn(F).astype(int) for n, fn, _ in RULES}, index=F.index)
    w = np.array([r[2] for r in RULES]); S = pd.DataFrame(index=F.index); S["rule_score"] = (rh.values * w).sum(1) / w.sum()

    def gb(cols):
        return CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=4, learning_rate=.08, max_iter=150, random_state=C.SEED),
                                      method="isotonic", cv=3).fit(F.loc[tr, cols], y[tr])
    onb = CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
                                 method="sigmoid", cv=3).fit(F.loc[tr, FEAT_ONB], y[tr])
    c_ml, c_mlg = FEAT_ONB + FEAT_BEH, FEAT_ONB + FEAT_BEH + FEAT_GRAPH
    c_nodoc = [c for c in c_mlg if c != "doc_mismatch"]
    m_ml, m_mlg, m_nd = gb(c_ml), gb(c_mlg), gb(c_nodoc)
    S["onboarding_p"] = onb.predict_proba(F[FEAT_ONB])[:, 1]; S["ml_p"] = m_ml.predict_proba(F[c_ml])[:, 1]
    S["mlg_p"] = m_mlg.predict_proba(F[c_mlg])[:, 1]; S["mlg_nodoc_p"] = m_nd.predict_proba(F[c_nodoc])[:, 1]

    sc = StandardScaler().fit(np.log1p(F.loc[tr, FEAT_BEH].clip(lower=0))); Xa = sc.transform(np.log1p(F[FEAT_BEH].clip(lower=0)))
    a1 = -IsolationForest(n_estimators=200, random_state=C.SEED).fit(Xa[tr]).score_samples(Xa)
    ae = MLPRegressor(hidden_layer_sizes=(8, 3, 8), max_iter=500, random_state=C.SEED).fit(Xa[tr & (y == 0)], Xa[tr & (y == 0)])
    a2 = ((ae.predict(Xa) - Xa) ** 2).mean(1)
    pct = lambda a: np.searchsorted(np.sort(a[tr]), a) / tr.sum()
    S["iso_pct"], S["ae_pct"] = pct(a1), pct(a2); S["anomaly"] = (S.iso_pct + S.ae_pct) / 2
    S["ens_rules_ml"] = .7 * S.ml_p + .3 * S.rule_score
    S["ens_full"] = .6 * S.mlg_p + .2 * S.rule_score + .2 * S.anomaly

    n_te = int(te.sum()); K = int(np.ceil(C.REVIEW_CAPACITY * n_te)); yt = y[te]
    var = [("Rule-only baseline", "rule_score", False), ("Onboarding LR only (KYC features)", "onboarding_p", True),
           ("Anomaly only (IsoForest+autoencoder)", "anomaly", False), ("ML (no graph)", "ml_p", True),
           ("Rules + ML ensemble (no graph)", "ens_rules_ml", True), ("ML + graph", "mlg_p", True),
           ("ML + graph - document features", "mlg_nodoc_p", True), ("Full ensemble (rules+ML+graph+anomaly)", "ens_full", True)]
    mt = pd.DataFrame([dict(model=n, score=c, **metrics_row(yt, S[c].values[te], K, p)) for n, c, p in var])
    mule_te = (ci.fraud_type.values == "mule")[te]
    mt["mule_recall_at_K"] = [mule_te[topk_idx(S[c].values[te], K)].sum() / mule_te.sum() for _, c, _ in var]

    thr_rev = np.sort(S.ens_full.values[te])[::-1][K - 1]
    thr_esc = np.sort(S.ens_full.values[te])[::-1][max(int(np.ceil(C.ESCALATE_SHARE * n_te)) - 1, 0)]
    tier = pd.Series(np.where(S.ens_full >= thr_esc, "ESCALATE", np.where(S.ens_full >= thr_rev, "REVIEW", "LOW")), index=F.index)
    pi = permutation_importance(m_mlg, F.loc[te, c_mlg], yt, scoring="average_precision", n_repeats=3, random_state=C.SEED)
    imp = pd.Series(pi.importances_mean, index=c_mlg).sort_values(ascending=False)
    cal, pr = {}, {}
    for nm, c in [("Onboarding LR", "onboarding_p"), ("ML (no graph)", "ml_p"), ("ML + graph", "mlg_p")]:
        a, b = calibration_curve(yt, S[c].values[te], n_bins=8, strategy="quantile"); cal[nm] = (b.tolist(), a.tolist())
    for nm, c in [("Rule-only", "rule_score"), ("Anomaly only", "anomaly"), ("ML (no graph)", "ml_p"), ("ML + graph", "mlg_p"), ("Full ensemble", "ens_full")]:
        p, r, _ = precision_recall_curve(yt, S[c].values[te]); pr[nm] = (r.tolist(), p.tolist())
    h1 = boot_diff(yt, S.ens_rules_ml.values[te], S.rule_score.values[te], K)
    psi = pd.DataFrame({"feature": c_mlg, "PSI_train_vs_test": [_psi(F.loc[tr, c].values, F.loc[te, c].values) for c in c_mlg]})
    return dict(S=S, tier=tier, rule_hits=rh, tr=tr, te=te, y=y, K=K, metrics=mt, importance=imp, calibration=cal, pr=pr, h1=h1, psi=psi,
                thr=(float(thr_rev), float(thr_esc)), reasoner_full=Reasoner(F, y, tr, c_mlg), reasoner_onb=Reasoner(F, y, tr, FEAT_ONB),
                models=dict(onb=onb, ml=m_ml, mlg=m_mlg), cols=dict(ml=c_ml, mlg=c_mlg))


def _psi(a, b, bins=10):
    q = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
    if len(q) < 3: return 0.0
    pa = np.histogram(a, bins=q)[0] / len(a) + 1e-6; pb = np.histogram(np.clip(b, q[0], q[-1]), bins=q)[0] / len(b) + 1e-6
    return float(((pa - pb) * np.log(pa / pb)).sum())


def scenario_alerts(tx, cust, te_ids):
    """Batch / replayable surveillance scenarios at transaction-day level, with per-customer de-duplication."""
    t = tx[tx.customer_id.isin(te_ids)].merge(cust[["customer_id", "declared_income", "device_id"]], on="customer_id", suffixes=("", "_home"))
    t["inc_m"] = t.declared_income / 12; out = []
    g = t[t.amount.between(9000, 9999.99)].groupby(["customer_id", "day"]).size().reset_index(name="n"); out.append(g[g.n >= 2].assign(scenario="Structuring day"))
    g = t.groupby(["customer_id", "day"]).size().reset_index(name="n"); out.append(g[g.n >= 10].assign(scenario="Velocity burst"))
    g = t[(t.hour < 6) & (t.amount >= 3 * t.inc_m * .1)].groupby(["customer_id", "day"]).size().reset_index(name="n"); out.append(g[g.n >= 2].assign(scenario="Night high-value"))
    g = t[(t.device_id != t.device_id_home) & (t.amount >= 15000)].groupby(["customer_id", "day"]).size().reset_index(name="n"); out.append(g.assign(scenario="New-device high-value"))
    return pd.concat(out, ignore_index=True)
