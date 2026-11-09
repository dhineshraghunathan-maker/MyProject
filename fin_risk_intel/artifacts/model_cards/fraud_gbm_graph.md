# Model card - Fraud / mule risk (calibrated gradient boosting + graph features)
**Purpose:** Customer-level transaction & fraud risk
**Intended use:** decision SUPPORT for an authorised investigator; never autonomous. **Out of scope:** onboarding rejection, freezing, payment blocking, regulatory filing.
**Data:** synthetic de-identified customers/transactions (seed 42); time-based split on account-opening date (train earlier 65%, test later 35%).
**Inputs:** behavioural aggregates, graph features, KYC features. Age, region and segment are NOT inputs (used only for fairness audit; segment is used for peer-group normalisation of average amount).
**Metrics (test, review capacity K=53):** PR-AUC 0.940; precision@K 1.000; recall@K 0.697; Brier 0.0132
**Explainability:** reason codes from an interpretable surrogate; permutation importance; linked sub-graph.
**Limitations:** synthetic data - results show method behaviour, not production performance; labels are delayed and noisy in reality; drift monitored via PSI.
