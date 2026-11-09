"""Feature engineering + entity resolution graph (customer-device-address-beneficiary)."""
import numpy as np
import pandas as pd
import networkx as nx
import config as C


def behaviour_features(cust, tx):
    t = tx.copy(); home = cust.set_index("customer_id").device_id
    t["near"] = t.amount.between(9000, 9999.99); t["night"] = t.hour < 6
    t["xb"] = t.country != "IN"; t["newdev"] = t.device_id != t.customer_id.map(home)
    g = t.groupby("customer_id")
    f = pd.DataFrame({"n_txn": g.size(), "total_amt": g.amount.sum(), "mean_amt": g.amount.mean(),
                      "std_amt": g.amount.std().fillna(0), "max_amt": g.amount.max(), "near_thr_cnt": g.near.sum(),
                      "night_share": g.night.mean(), "n_ben": g.beneficiary_id.nunique(),
                      "new_dev_share": g.newdev.mean(), "xborder_share": g.xb.mean()})
    f["max_daily_cnt"] = t.groupby(["customer_id", "day"]).size().groupby("customer_id").max()
    pv = t.groupby(["customer_id", "day", "direction"]).amount.sum().unstack(fill_value=0)
    f["pass_through"] = np.minimum(pv.credit, pv.debit).groupby("customer_id").sum() / (pv.credit.groupby("customer_id").sum() + 1)
    ci = cust.set_index("customer_id").loc[f.index]
    f["max_to_income"] = f.max_amt / (ci.declared_income / 12)
    f["peer_dev"] = (f.mean_amt - f.groupby(ci.segment).mean_amt.transform("mean")) / (f.groupby(ci.segment).mean_amt.transform("std") + 1e-9)
    return f.fillna(0)


def build_graph(cust, tx, max_deg=12):
    """Customer projection graph: edge if two customers share a device / address / beneficiary."""
    a = lambda df, c: df[["customer_id", c]].rename(columns={c: "res"})
    pairs = pd.concat([a(cust, "device_id"), a(cust, "address_id"), a(tx, "device_id"), a(tx, "beneficiary_id")]).drop_duplicates()
    deg = pairs.groupby("res").customer_id.nunique()
    keep = deg[(deg >= 2) & (deg <= max_deg)].index
    P = nx.Graph(); P.add_nodes_from(cust.customer_id)
    for res, grp in pairs[pairs.res.isin(keep)].groupby("res"):
        cs = list(grp.customer_id)
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                if P.has_edge(cs[i], cs[j]): P[cs[i]][cs[j]]["w"] += 1; P[cs[i]][cs[j]]["res"].append(res)
                else: P.add_edge(cs[i], cs[j], w=1, res=[res])
    comp = {n: len(c) for c in nx.connected_components(P) for n in c}
    comm = {n: len(c) for c in nx.community.louvain_communities(P, seed=C.SEED) for n in c}
    gf = pd.DataFrame({"g_degree": dict(P.degree()), "g_wdegree": dict(P.degree(weight="w")), "g_component": comp,
                       "g_pagerank": nx.pagerank(P, weight="w"), "g_community": comm})
    return gf, P


def build_features(cust, tx, kyc_r):
    bf = behaviour_features(cust, tx); gf, P = build_graph(cust, tx)
    ci = cust.set_index("customer_id")
    F = bf.join(gf).join(ci[["occ_risk", "country_risk"]]).join(kyc_r[["doc_mismatch", "name_sim", "dob_match", "addr_sim"]])
    F["log_income"] = np.log(ci.declared_income)
    return F.loc[cust.customer_id], P
