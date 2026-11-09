# Model card - Onboarding CDD risk (calibrated logistic regression)
**Purpose:** Customer due-diligence risk at account opening
**Intended use:** decision SUPPORT for an authorised investigator; never autonomous. **Out of scope:** onboarding rejection, freezing, payment blocking, regulatory filing.
**Data:** synthetic de-identified customers/transactions (seed 42); time-based split on account-opening date (train earlier 65%, test later 35%).
**Inputs:** occupation risk, country risk, document mismatch, log income. Age, region and segment are NOT inputs (used only for fairness audit; segment is used for peer-group normalisation of average amount).
**Metrics (test, review capacity K=53):** PR-AUC 0.196; precision@K 0.321; recall@K 0.224; Brier 0.0643
**Explainability:** reason codes from an interpretable surrogate; permutation importance; linked sub-graph.
**Limitations:** synthetic data - results show method behaviour, not production performance; labels are delayed and noisy in reality; drift monitored via PSI.
