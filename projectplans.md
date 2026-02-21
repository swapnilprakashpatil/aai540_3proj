Data Sources:

CMS Open Payments Program Year 2024 public dataset (general payments focus) (CMS,
2025a). https://openpaymentsdata.cms.gov/datasets/download

• CMS summary: ~16.16M records totaling $13.18B for PY2024 (CMS, 2025a).
• Project extract: 15,397,627 rows × 91 columns (computed after ingestion).

Why this dataset?
• Real-world healthcare compliance/ethics transparency domain
• Large scale supports realistic pipeline engineering
• Annual refresh cycle supports drift + retraining storyline

Risks
• Potential PII exposure (provider identity/location); minimize use of direct identifiers in
modeling.
• Interpretation risk: outputs must be framed as “unusual patterns,” consistent with CMS
transparency guidance.

5
Data Engineering:
Storage
• s3://opayments-raw/ — raw downloads
• s3://opayments-curated/ — cleaned parquet (partitioned)
• s3://opayments-features/ — feature tables for training/scoring
• s3://opayments-preds/ — scored outputs

Preprocessing  
• CSV → parquet conversion + partitioning
• Type casts + standardization
• Missing value handling
• Deduplication by record identifier fields per CMS data definitions.
Training Data:
Split strategy
Time-based split:
• Train on early months (or prior year)
• Validate on middle months
• Test on later months (or next year)

Labeling techniques
Weak evaluation signals from publication metadata such as changed vs unchanged records
Feature Engineering:
Fields to use / exclude
Use: amount, payment date, nature/form, reporting entity identifiers, specialty/taxonomy, state,
recipient type.
Exclude: provider names and free text fields; avoid features that personalize to individuals.

Combinations / bucketing
• Aggregate to recipient-month
• Amount of log transforms
• Peer group normalization (specialty + state + recipient type)

Transformations

• log1p(amount)
• robust scaling (median/IQR)
• limited encoding for high-cardinality categories (frequency encoding)

6
Model Training & Evaluation:
Training method
Train Isolation Forest on aggregated feature table; tune contamination level to match review
capacity (e.g., top 0.5–2%).

Algorithm
Isolation Forest + baseline robust peer outlier scoring.
Key parameters (initial)
• n_estimators: 200–500
• contamination: 0.005–0.02
• max_samples: 256 or auto
• max_features: 0.7–1.0
Evaluation
• Top K utility + stability checks
• Drift checks on features and scores
• Manual review of reason codes for top anomalies

Model Deployment:

7
Instance size
Small CPU instances for processing/training/batch scoring (e.g., m5. large) to fit $50 credits.

Batch or real time
Batch only (monthly/on-demand). This avoids always-on endpoint costs and matches the
publication cadence.
Model Monitoring:
Model monitoring
• anomaly rate drift
• score distribution drift
• reason-code distribution drift
Infrastructure monitoring
• job failure alarms
• runtime anomalies
• S3 input/output completeness checks
Data monitoring
• schema drift
• Missingness drift
• feature distribution drift (amounts, category mix, payer diversity)

Model CI/CD:
Checkpoints
• lint + unit tests
• schema tests
• pipeline integration test on sampled data
• train + evaluate gate
• register model + approval
• batch scoring job post-approval
Tests
• schema validation
• feature quality checks (ranges/missingness)
• evaluation gates (stability + anomaly rate bounds)
• security checks (IAM least privilege, S3 encryption)

Security Checklist, Privacy and Other Risks:

• PHI: No

8
• PII: Yes (provider identity/location). Justification: comes from public dataset; mitigation:
encrypt storage, restrict access, do not use names as features, and present results as
“review prioritization,” not wrongdoing claims.
• User behavior tracked: No
• Credit card info: No
• S3 buckets: raw/curated/features/preds (as listed)
• Bias considerations: differing payment patterns across specialties/regions may be
legitimate; use peer-group comparisons and subgroup monitoring.
• Ethical concerns: outputs can be misused or misinterpreted; align wording and
documentation with CMS guidance on interpretation.

9
Future Enhancements:

1. Add multi-level scoring (recipient-month + company-month + specialty-state
   benchmarks).
2. Add semi-supervised learning from reviewer feedback (“expected/unexpected”) to
   improve precision.
3. Improve explanations (e.g., SHAP on a supervised model trained from pseudo-labels).
4. Extend to Research Payments and Ownership/Investment datasets (separate pipelines).
5. Add automated data quality rules (missingness anomalies, schema changes across
   program years).
   References
   Centers for Medicare & Medicaid Services. (2025a). Open Payments: Program overview and
   data updates (Program Year 2024 publication). Open Payments.
   https://openpaymentsdata.cms.gov/datasets/download  
   Centers for Medicare & Medicaid Services. (2025b). Open Payments data dictionary /
   methodology documentation for public use files. Open Payments.
   https://openpaymentsdata.cms.gov/dataset/e6b17c6a-2534-4207-a4a1-6746a14911ff#data-
   dictionary
