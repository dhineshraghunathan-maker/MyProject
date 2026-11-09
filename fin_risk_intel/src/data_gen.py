"""Synthetic de-identified data: customers, transactions, KYC documents (canonical data model)."""
import numpy as np
import pandas as pd
import config as C

FIRST = ["Arun", "Priya", "Karthik", "Divya", "Suresh", "Meena", "Rahul", "Anita", "Vikram", "Lakshmi",
         "Imran", "Sneha", "Rohan", "Kavya", "Manoj", "Pooja", "Naveen", "Deepa", "Sanjay", "Ritu"]
LAST = ["Kumar", "Sharma", "Iyer", "Reddy", "Nair", "Singh", "Patel", "Das", "Menon", "Gupta",
        "Khan", "Rao", "Joshi", "Pillai", "Verma"]
SEG, REG = ["Retail", "Student", "Senior", "SME"], ["North", "South", "East", "West"]
ID_TYPES = ["Passport", "National ID", "Driving License", "Voter ID"]
INCOME_SOURCES = ["Salaried employment", "Business ownership", "Freelance / gig work", "Pension", "Rental income", "Family support"]
INCOME_RANGES = ["Below ₹2.5L", "₹2.5L - ₹5L", "₹5L - ₹10L", "₹10L - ₹25L", "₹25L - ₹50L", "₹50L+"]
OCC_RISK = {"Salaried": .1, "Teacher": .1, "Engineer": .1, "Retired": .1, "Student": .3,
            "Freelancer": .4, "Trader": .5, "Cash business": .8}
CITY = {"North": "Delhi", "South": "Chennai", "East": "Kolkata", "West": "Mumbai"}
STREET = ["MG Road", "Anna Salai", "Lake View", "Park Street", "Temple Road", "Gandhi Nagar"]


def _perturb(s, rng, k):
    s = list(s)
    for _ in range(k):
        p = int(rng.integers(len(s))); op = int(rng.integers(3))
        if op == 0: s[p] = chr(int(rng.integers(97, 123)))
        elif op == 1 and len(s) > 3: del s[p]
        else: s.insert(p, chr(int(rng.integers(97, 123))))
    return "".join(s)


def generate():
    rng = np.random.default_rng(C.SEED); n = C.N_CUSTOMERS
    seg = rng.choice(SEG, n, p=[.55, .15, .15, .15]); region = rng.choice(REG, n, p=[.3, .3, .2, .2])
    age = np.where(seg == "Student", rng.integers(18, 26, n),
                   np.where(seg == "Senior", rng.integers(60, 85, n), rng.integers(24, 60, n)))
    occ = np.array([{"Student": "Student", "Senior": "Retired"}.get(s, None) or
                    rng.choice(["Trader", "Cash business"] if s == "SME" else ["Salaried", "Teacher", "Engineer", "Freelancer"])
                    for s in seg])
    income = rng.lognormal(13.2, .5, n) * np.where(seg == "Student", .25, np.where(seg == "SME", 2.0, 1.0))
    open_day = rng.integers(0, 540, n)

    ft = np.array(["none"] * n, dtype=object); idx = rng.permutation(n)
    n_mule, n_ato, n_scam = int(.03 * n), int(.02 * n), int(.02 * n)
    mule_idx, ato_idx, scam_idx = idx[:n_mule], idx[n_mule:n_mule + n_ato], idx[n_mule + n_ato:n_mule + n_ato + n_scam]
    ft[mule_idx], ft[ato_idx], ft[scam_idx] = "mule", "ato", "scam"
    ring = -np.ones(n, int)
    for r, chunk in enumerate(np.array_split(mule_idx, 12)): ring[chunk] = r

    dev = [f"D{i}" for i in range(n)]; addr = [f"A{i}" for i in range(n)]
    for i in mule_idx:
        if rng.random() < .60: dev[i] = f"DR{ring[i]}"
        if rng.random() < .55: addr[i] = f"AR{ring[i]}"
    legit = np.where(ft == "none")[0]
    for a, b in rng.choice(legit, size=2 * int(.04 * n), replace=False).reshape(-1, 2):   # households
        dev[b], addr[b] = dev[a], addr[a]

    cols = ["customer_id", "day", "hour", "amount", "direction", "beneficiary_id", "device_id", "country", "is_fraud_txn"]
    acc = {c: [] for c in cols}
    for i in range(n):
        cid = f"C{i:05d}"; inc_m = income[i] / 12; k = int(rng.integers(25, 60)); st = int(open_day[i]) + 1
        day = rng.integers(st, st + 150, k); hour = np.clip(rng.normal(14, 4, k), 0, 23).astype(int)
        amt = inc_m * .03 * rng.lognormal(0, .8, k)
        direc = np.where(rng.random(k) < .6, "debit", "credit").astype(object)
        own = [f"B{i}_{j}" for j in range(int(rng.integers(3, 7)))]
        if rng.random() < .35: own[0] = f"F{int(rng.integers(400))}"     # friends/family payees create benign graph links
        ben = np.array([own[int(rng.integers(len(own)))] if rng.random() < .7 else f"M{int(rng.integers(25))}"
                        for _ in range(k)], dtype=object)
        dv = np.full(k, dev[i], dtype=object); ct = np.where(rng.random(k) < .02, "XA", "IN").astype(object)
        fl = np.zeros(k, bool); ex = None; t = ft[i]
        if t == "none":                                    # benign noise that fools fixed rules
            if rng.random() < .10: amt[int(rng.integers(k))] = inc_m * rng.uniform(3, 6)
            if rng.random() < .09: amt[rng.choice(k, 3, replace=False)] = rng.uniform(9000, 9990, 3)
            if rng.random() < .06: hour[rng.choice(k, 4, replace=False)] = rng.integers(0, 6, 4)
            if rng.random() < .03: dv[rng.choice(k, 3, replace=False)] = f"DN{i}"
            if rng.random() < .07:
                m = int(rng.integers(10, 14)); d0 = int(rng.integers(st, st + 140))
                ex = dict(day=np.full(m, d0), hour=rng.integers(9, 20, m), amt=inc_m * .02 * rng.lognormal(0, .5, m),
                          direc=np.array(["debit"] * m, dtype=object),
                          ben=np.array([f"M{int(rng.integers(25))}" for _ in range(m)], dtype=object),
                          dv=np.full(m, dev[i], dtype=object), ct=np.full(m, "IN", dtype=object), fl=np.zeros(m, bool))
        elif t == "mule":
            stealth = rng.random() < .45; m = int(rng.integers(4, 8) if stealth else rng.integers(12, 20)); d0 = int(rng.integers(st, st + 120)); h = m // 2
            ex = dict(day=d0 + rng.integers(0, 3 if not stealth else 12, m),
                      hour=np.where(rng.random(m) < (.1 if stealth else .5), rng.integers(0, 6, m), rng.integers(8, 22, m)),
                      amt=np.where(rng.random(m) < (.1 if stealth else .6), rng.uniform(9000, 9990, m), inc_m * rng.uniform(.05, .3, m)),
                      direc=np.array(["credit"] * h + ["debit"] * (m - h), dtype=object),
                      ben=np.array([f"P{int(rng.integers(9000))}" for _ in range(h)] +
                                   [(f"BR{ring[i]}_{int(rng.integers(2))}" if rng.random() < .55 else f"BX{i}_{int(rng.integers(3))}") for _ in range(m - h)], dtype=object),
                      dv=np.full(m, dev[i], dtype=object), ct=np.where(rng.random(m) < (.03 if stealth else .15), "XA", "IN").astype(object),
                      fl=np.ones(m, bool))
        elif t == "ato":
            stealth = rng.random() < .45; m = int(rng.integers(3, 6) if stealth else rng.integers(6, 15)); d0 = st + int(rng.integers(20, 100))
            ex = dict(day=d0 + rng.integers(0, 3, m),
                      hour=np.where(rng.random(m) < (.15 if stealth else .6), rng.integers(0, 6, m), rng.integers(8, 23, m)),
                      amt=inc_m * (rng.uniform(.1, .6, m) if stealth else rng.uniform(.5, 2, m)), direc=np.array(["debit"] * m, dtype=object),
                      ben=np.array([f"BN{i}_{j}" for j in range(m)], dtype=object),
                      dv=np.full(m, f"DN{i}", dtype=object), ct=np.where(rng.random(m) < (.05 if stealth else .3), "XA", "IN").astype(object),
                      fl=np.ones(m, bool))
        else:                                              # scam victim
            m = int(rng.integers(1, 3)); d0 = st + int(rng.integers(10, 120))
            ex = dict(day=d0 + rng.integers(0, 5, m), hour=rng.integers(8, 22, m), amt=inc_m * rng.uniform(1.5, 8, m),
                      direc=np.array(["debit"] * m, dtype=object),
                      ben=np.array([f"BS{int(rng.integers(40))}" for _ in range(m)], dtype=object),
                      dv=np.full(m, dev[i], dtype=object), ct=np.full(m, "IN", dtype=object), fl=np.ones(m, bool))
        if ex:
            day, hour, amt = np.concatenate([day, ex["day"]]), np.concatenate([hour, ex["hour"]]), np.concatenate([amt, ex["amt"]])
            direc, ben = np.concatenate([direc, ex["direc"]]), np.concatenate([ben, ex["ben"]])
            dv, ct, fl = np.concatenate([dv, ex["dv"]]), np.concatenate([ct, ex["ct"]]), np.concatenate([fl, ex["fl"]])
        for c, v in zip(cols, [np.full(len(day), cid, dtype=object), day, hour, amt.round(2), direc, ben, dv, ct, fl]):
            acc[c].append(v)
    tx = pd.DataFrame({c: np.concatenate(v) for c, v in acc.items()})
    tx["date"] = pd.Timestamp("2025-01-01") + pd.to_timedelta(tx.day, unit="D")
    tx["channel"] = np.random.default_rng(C.SEED + 1).choice(["UPI", "NEFT", "Card", "ATM"], len(tx))
    tx.insert(0, "txn_id", [f"T{j:07d}" for j in range(len(tx))])

    names = [f"{rng.choice(FIRST)} {rng.choice(LAST)}" for _ in range(n)]
    id_types = np.array([rng.choice(ID_TYPES) for _ in range(n)], dtype=object)
    sources = np.array([rng.choice(INCOME_SOURCES) for _ in range(n)], dtype=object)
    income_ranges = np.array([rng.choice(INCOME_RANGES) for _ in range(n)], dtype=object)
    pep = np.array([rng.choice(["Yes", "No"], p=[.08, .92]) for _ in range(n)], dtype=object)

    cust = pd.DataFrame(dict(customer_id=[f"C{i:05d}" for i in range(n)], name=names, age=age, segment=seg, region=region,
                             occupation=occ, declared_income=income.round(0), source_of_income=sources,
                             income_range=income_ranges, pep=pep, id_type=id_types, open_day=open_day, device_id=dev,
                             address_id=addr, ring_id=ring, fraud_type=ft, is_fraud=(ft != "none").astype(int)))
    cust["age_band"] = pd.cut(cust.age, [0, 24, 39, 59, 120], labels=["<25", "25-39", "40-59", "60+"]).astype(str)
    cust["occ_risk"] = cust.occupation.map(OCC_RISK)
    cust["country_risk"] = (rng.random(n) < np.where(ft == "mule", .25, .03)).astype(int)

    addr_text = {a: f"{int(rng.integers(1, 300))}, {STREET[int(rng.integers(len(STREET)))]}, {CITY[r]}"
                 for a, r in zip(addr, region)}
    rows = []
    for i in range(n):
        dob = f"{2026 - age[i]}-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 29)):02d}"
        bad = rng.random() < (.45 if ft[i] == "mule" else .04)
        dn, dd, da = names[i], dob, addr_text[addr[i]]
        doc_id_type = id_types[i]
        doc_id_number = f"ID{int(rng.integers(10**7, 10**8))}"
        doc_source_of_income = sources[i]
        doc_income_range = income_ranges[i]
        doc_pep = pep[i]
        doc_occupation = occ[i]
        if bad:
            dn = _perturb(names[i], rng, int(rng.integers(2, 4)))
            if rng.random() < .5: dd = f"{int(dob[:4]) + int(rng.integers(1, 4))}{dob[4:]}"
            if rng.random() < .6: da = _perturb(da, rng, 2)
            if rng.random() < .4: doc_id_type = rng.choice(ID_TYPES)
            if rng.random() < .3: doc_id_number = f"ID{int(rng.integers(10**7, 10**8))}"
            if rng.random() < .3: doc_occupation = rng.choice(["Salaried", "Teacher", "Engineer", "Freelancer", "Trader", "Cash business", "Student", "Retired"])
            if rng.random() < .3: doc_source_of_income = rng.choice(INCOME_SOURCES)
            if rng.random() < .3: doc_income_range = rng.choice(INCOME_RANGES)
            if rng.random() < .25: doc_pep = rng.choice(["Yes", "No"])
        rows.append(dict(customer_id=f"C{i:05d}", declared_name=names[i], declared_dob=dob, declared_address=addr_text[addr[i]],
                         declared_occupation=occ[i], declared_source_of_income=sources[i], declared_income_range=income_ranges[i],
                         declared_pep=pep[i], declared_id_type=id_types[i], declared_id_number=f"ID{int(rng.integers(10**7, 10**8))}",
                         doc_name=dn, doc_dob=dd, doc_address=da, doc_occupation=doc_occupation, doc_source_of_income=doc_source_of_income,
                         doc_income_range=doc_income_range, doc_pep=doc_pep, doc_id_type=doc_id_type, doc_id_number=doc_id_number,
                         id_number=doc_id_number, id_type=doc_id_type))
    return dict(customers=cust, transactions=tx, kyc=pd.DataFrame(rows))


def data_dictionary():
    r = [("Customer", "customer_id", "str", "Pseudonymous customer key"), ("Customer", "segment/region/age_band", "str", "Used ONLY for fairness audit and peer grouping, never as model inputs (except segment peer-group z-score)"),
         ("Customer", "occupation, occ_risk", "str/float", "Declared occupation and rule-based occupation risk"), ("Customer", "declared_income, source_of_income, income_range, pep, id_type", "mixed", "Synthetic profile fields for onboarding and compliance review"),
         ("Customer", "country_risk", "int", "1 if nationality/residence linked to a high-risk jurisdiction"), ("Customer", "ring_id, fraud_type, is_fraud", "mixed", "Investigation OUTCOME (ground truth label; delayed in reality)"),
         ("Account", "account_id", "str", "One account per customer (AC + customer_id)"), ("Document", "doc_name, doc_dob, doc_address, id_type, id_number, doc_occupation, doc_source_of_income, doc_income_range, doc_pep", "str", "Fields read from the identity document / KYC form"),
         ("Document", "declared_*", "str", "Fields typed on the application"), ("Document", "doc_mismatch", "float", "1 - weighted similarity(declared, document)"),
         ("Device", "device_id", "str", "Home device (customers.device_id) or device observed on transaction"), ("Beneficiary", "beneficiary_id", "str", "Payee / payer identifier; M* = merchants (high degree, ignored in graph)"),
         ("Transaction", "txn_id, customer_id, date, hour, amount, direction, channel, country", "mixed", "Debit/credit event"), ("Transaction", "is_fraud_txn", "bool", "Synthetic ground truth for injected fraud events"),
         ("Alert", "customer_id, tier, ens_full, rule_hits, reasons", "mixed", "Prioritised alert produced by decisioning (artifacts/alerts.csv)"),
         ("Outcome", "is_fraud, fraud_type", "mixed", "Confirmed outcome used for time-based evaluation")]
    return pd.DataFrame(r, columns=["entity", "fields", "type", "description"])
