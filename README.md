# CMS Open Payments Anomaly Detection

**AAI-540 Machine Learning Operations - Final Team Project**

## 🔗 Project Links

- **Interactive Notebook:** https://swapnilprakashpatil.github.io/aai540_3proj/
- **Live Demo:** http://cms-anomaly-detection-frontend-prod.s3-website-us-east-1.amazonaws.com/

## Team Members

- Swapnil Patil
- Jamshed Nabizada
- Tej Bahadur Singh

## Professor:

- Sean Coyne

## Project Scope

### Project Background

The CMS Open Payments program publishes information about financial relationships between drug/medical device companies ("reporting entities") and healthcare providers ("covered recipients") to promote transparency. These relationships can include payments for items such as meals, travel, gifts, speaking fees, and research-related transfers of value. The published data is open to interpretation and does not inherently indicate an improper relationship (Centers for Medicare & Medicaid Services [CMS], 2025a).

This project implements an end-to-end MLOps system that assigns risk scores to Open Payments records to prioritize statistically unusual payment patterns for compliance review. The system is designed for triage/prioritization and does not label records as fraud. The ML problem is framed as unsupervised anomaly detection on large-scale tabular data.

### Technical Approach

#### Dataset

**Primary dataset:** CMS Open Payments Program Year 2024 General Payments

- Published June 30, 2025
- Covers payments from January 1 - December 31, 2024
- Approximately 16.16 million records totaling $13.18B in payments
- Project extract: 15,397,627 rows × 91 columns

The dataset's scale reinforces the need for cloud infrastructure, parquet conversion, partitioning, and batch processing capabilities.

#### Model Architecture

The project implements and compares three anomaly detection approaches:

1. **Isolation Forest (Primary Model)**
   - Fast, stable, and produces interpretable "unusualness" scores
   - Aligns well with risk signals: high amounts, weekend activity, unusual payment patterns
   - Hyperparameter-optimized to reduce false positive rate
   - Selected as the primary model for production deployment

2. **Autoencoder (Neural Network)**
   - Deep learning approach for capturing complex patterns
   - Architecture: 50-epoch training with early stopping
   - Reconstruction error-based anomaly detection
   - Effective at identifying rare feature combinations

3. **XGBoost (Supervised Pseudo-labeling)**
   - Gradient boosting approach trained on pseudo-labels
   - Near-perfect AUC on validation data
   - Used as complementary model for high-confidence predictions

#### Key Features

The models learn "unusualness" from:

- Payment amount aggregates (sum, mean, max, standard deviation)
- Payment frequency and temporal patterns
- Diversity metrics (distinct reporting entities per recipient)
- Payment nature and form categorical distributions
- Peer deviations by specialty, state, and recipient type
- Historical comparison features (payment trends over time)

#### Model Evaluation

Because Open Payments lacks ground-truth fraud labels, evaluation focuses on:

- **Top-K Utility:** Highest-ranked anomalies represent truly unusual patterns
- **Temporal Stability:** Score distributions remain consistent across time periods
- **Drift Detection:** Monitoring for data and concept drift
- **Interpretability:** Reason codes explain why records were flagged

### Project Results

#### Model Performance

- **Isolation Forest:** Achieved optimal contamination rate of 1-2% with stable anomaly detection across training and validation sets
- **Autoencoder:** Successfully captured reconstruction errors with clear separation between normal and anomalous patterns
- **XGBoost:** 95%+ AUC on pseudo-labeled validation data, demonstrating strong pattern recognition

#### Key Findings

The EDA and model outputs reveal:

- Most payments are small (typical value ~$20), but a small fraction shows extreme values
- Clear payment distribution patterns by specialty, state, and recipient type
- Temporal seasonality aligned with annual reporting cycles
- Data quality issues requiring validation (invalid state codes, payment nature encoding)

#### Production Model

The **Isolation Forest** model was selected for production deployment because:

- Fast inference suitable for batch processing
- Stable performance across train/test splits
- Interpretable scores aligned with business risk indicators
- Lower computational requirements fit AWS credit constraints
- Hyperparameter tuning reduced anomaly rate to actionable levels (~1-2%)

The Autoencoder and XGBoost models serve as complementary validators for high-risk cases requiring additional review.

### Goals vs Non-Goals

| Goals                                                                                                             | Non-Goals                                                                           |
| ----------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Build an end-to-end AWS ML workflow: ingest → preprocess → feature engineering → train → registry → batch scoring | Real-time streaming ingestion or real-time detection                                |
| Generate interpretable risk scores and top K anomaly lists with "reason codes"                                    | Automated enforcement actions or definitive fraud claims                            |
| Add monitoring for data drift and anomaly-rate drift; retrain on schedule or drift thresholds                     | Production UI: outputs are batch files/tables and demo artifacts                    |
| Keep AWS within limited credits using batch inference and small CPU jobs                                          | Linking external datasets to infer protected attributes or intent                   |
| Maintain correct program framing: unusual pattern detection, not wrongdoing/fraud determination                   | Maximizing state-of-the-art anomaly methods at the cost of simplicity and stability |

---

## Solution Overview

The system ingests the PY2024 Open Payments general payments data into S3, converts it to curated parquet, engineers recipient-month aggregate features, trains an anomaly detection model, and produces batch risk scores. Monitoring checks for feature distribution drift and score/anomaly-rate drift, and retraining is triggered on schedule or drift thresholds. This batch-first design aligns well with the Open Payments annual publication lifecycle and refresh model (CMS, 2025a).

### Data Sources

**CMS Open Payments Program Year 2024 public dataset (general payments focus)** (CMS, 2025a)  
https://openpaymentsdata.cms.gov/datasets/download

- CMS summary: ~16.16M records totaling $13.18B for PY2024 (CMS, 2025a)
- Project extract: 15,397,627 rows × 91 columns (computed after ingestion)

#### Why this dataset?

- Real-world healthcare compliance/ethics transparency domain
- Large scale supports realistic pipeline engineering
- Annual refresh cycle supports drift + retraining storyline

#### Risks

- Potential PII exposure (provider identity/location); minimize use of direct identifiers in modeling
- Interpretation risk: outputs must be framed as "unusual patterns," consistent with CMS transparency guidance

---

## Data Engineering

### Storage

- `s3://opayments-raw/` — raw downloads
- `s3://opayments-curated/` — cleaned parquet (partitioned)
- `s3://opayments-features/` — feature tables for training/scoring
- `s3://opayments-preds/` — scored outputs

### Preprocessing

- CSV → parquet conversion + partitioning
- Type casts + standardization
- Missing value handling
- Deduplication by record identifier fields per CMS data definitions

---

## Training Data

### Split strategy

**Time-based split:**

- Train on early months (or prior year)
- Validate on middle months
- Test on later months (or next year)

### Labeling techniques

Weak evaluation signals from publication metadata such as changed vs unchanged records

---

## Feature Engineering

### Fields to use / exclude

**Use:** amount, payment date, nature/form, reporting entity identifiers, specialty/taxonomy, state, recipient type

**Exclude:** provider names and free text fields; avoid features that personalize to individuals

### Combinations / bucketing

- Aggregate to recipient-month
- Amount of log transforms
- Peer group normalization (specialty + state + recipient type)

### Transformations

- `log1p(amount)`
- Robust scaling (median/IQR)
- Limited encoding for high-cardinality categories (frequency encoding)

---

## Model Training & Evaluation

### Training method

Train Isolation Forest on aggregated feature table; tune contamination level to match review capacity (e.g., top 0.5–2%).

### Algorithm

Isolation Forest + baseline robust peer outlier scoring

### Key parameters (initial)

- `n_estimators`: 200–500
- `contamination`: 0.005–0.02
- `max_samples`: 256 or auto
- `max_features`: 0.7–1.0

### Evaluation

- Top K utility + stability checks
- Drift checks on features and scores
- Manual review of reason codes for top anomalies

---

## Model Deployment

### Instance size

Small CPU instances for processing/training/batch scoring (e.g., `m5.large`) to fit $50 credits

### Batch or real time

**Batch only** (monthly/on-demand). This avoids always-on endpoint costs and matches the publication cadence.

---

## Model Monitoring

### Model monitoring

- Anomaly rate drift
- Score distribution drift
- Reason-code distribution drift

### Infrastructure monitoring

- Job failure alarms
- Runtime anomalies
- S3 input/output completeness checks

### Data monitoring

- Schema drift
- Missingness drift
- Feature distribution drift (amounts, category mix, payer diversity)

---

## Model CI/CD

### Checkpoints

- Lint + unit tests
- Schema tests
- Pipeline integration test on sampled data
- Train + evaluate gate
- Register model + approval
- Batch scoring job post-approval

### Tests

- Schema validation
- Feature quality checks (ranges/missingness)
- Evaluation gates (stability + anomaly rate bounds)
- Security checks (IAM least privilege, S3 encryption)

---

## References

Centers for Medicare & Medicaid Services. (2025a). _Open Payments: Program overview and data updates (Program Year 2024 publication)_. Open Payments. https://openpaymentsdata.cms.gov/datasets/download

Centers for Medicare & Medicaid Services. (2025b). _Open Payments data dictionary / methodology documentation for public use files_. Open Payments. https://openpaymentsdata.cms.gov/dataset/e6b17c6a-2534-4207-a4a1-6746a14911ff#data-dictionary

---

## MLOps Architecture and Implementation

This project implements a comprehensive MLOps pipeline covering the complete machine learning lifecycle from data ingestion to production monitoring.

### End-to-End Pipeline

#### 1. Data Engineering

**AWS S3 Data Lake Architecture:**

- `s3://opayments-raw/` — Raw CMS dataset downloads
- `s3://opayments-curated/` — Cleaned and partitioned parquet files
- `s3://opayments-features/` — Engineered features for training/scoring
- `s3://opayments-preds/` — Model predictions and anomaly scores

**AWS Athena Integration:**

- SQL-based querying directly on S3 data lake
- Partition pruning for optimized query performance
- Schema evolution support for annual CMS updates

**Data Processing Pipeline:**

1. CSV to Parquet conversion with column type enforcement
2. Partitioning by program year and month for query optimization
3. Data validation and quality checks
4. Missing value handling and deduplication
5. Feature engineering with historical aggregations

#### 2. Model Training Workflow

**Approach:**

- Time-based train/validation/test split to prevent leakage
- Recipient-month level aggregation for feature engineering
- Robust scaling using median/IQR normalization
- Peer-group comparisons (specialty × state × recipient type)

**Hyperparameter Optimization:**

- Grid Search: Systematic exploration of parameter space
- Randomized Search: Efficient broader exploration (30 iterations)
- Cross-validation with custom anomaly detection metric
- Optimal parameters selected based on validation score

**Model Artifacts:**

- Trained model (joblib serialized)
- Preprocessing objects (scalers, encoders)
- Feature names and configuration
- Anomaly threshold parameters
- Metadata (training date, dataset version, metrics)

#### 3. Model Registry and Versioning

**SageMaker Model Registry:**

- Model package groups for version management
- Metadata tracking: parameters, metrics, training dataset
- Approval workflow for production promotion
- Model lineage for reproducibility

**Model Card Generation:**

- Model purpose and intended use
- Training details and hyperparameters
- Evaluation results and limitations
- Bias considerations and monitoring requirements

#### 4. Production Deployment

**Amazon SageMaker Endpoint:**

- Real-time inference capability
- PyTorch container with custom inference script
- Instance type: ml.m5.large (cost-optimized for $50 credit limit)
- Auto-scaling configuration for variable load
- Data capture enabled for monitoring (100% sampling)

**Inference Pipeline:**

```python
Input → Preprocessing → Feature Engineering → Model Scoring → Post-processing → Output
```

**Deployment Validation:**

- Sample predictions compared against notebook outputs
- Anomaly rate validation (expected: 1-2%)
- Score distribution consistency checks
- Response time and throughput testing

#### 5. Model Monitoring

**Data Quality Monitoring:**

- **Feature Drift Detection:** Statistical tests for distribution changes
- **Schema Validation:** Column presence and type checking
- **Missing Value Alerts:** Threshold-based alerts for data quality issues
- **Categorical Drift:** New/invalid category detection

**Model Performance Monitoring:**

- **Anomaly Rate Drift:** Tracking changes in detection rate over time
- **Score Distribution Monitoring:** Percentile-based drift detection
- **Prediction Latency:** Response time tracking
- **Error Rate Monitoring:** Failed prediction tracking

**SageMaker Model Monitor:**

- Baseline statistics generation from production traffic
- Scheduled monitoring jobs (daily execution via Cron)
- Constraint violation detection
- Integration with CloudWatch for alerting

**CloudWatch Alarms:**

- Payment amount distribution drift (threshold: 15%)
- Feature baseline violation alerts
- Endpoint health and availability monitoring
- Custom metrics for business KPIs

#### 6. Automated Retraining

**Trigger Conditions:**

- Scheduled monthly retraining (align with CMS data updates)
- Feature drift exceeds threshold (>15% deviation)
- Anomaly rate drift (>25% change from baseline)
- Manual trigger for urgent updates

**Retraining Workflow:**

1. Fetch latest curated data from S3
2. Generate fresh features with updated historical context
3. Train new model with existing hyperparameters
4. Validate against holdout test set
5. Register new model version in Model Registry
6. Deploy to staging for validation
7. Promote to production upon approval

---

## CI/CD Pipeline Architecture

### GitHub Actions Workflows

#### 1. Terraform Infrastructure CI/CD

```yaml
Trigger: Push to main, Pull requests
Steps:
  - Terraform format validation
  - Terraform initialization
  - Terraform plan generation
  - Security scanning (checkov)
  - Terraform apply (main branch only)
```

#### 2. Lambda Function CI/CD

```yaml
Trigger: Changes in lambda/**
Steps:
  - Python lint (pylint, flake8)
  - Unit test execution
  - Dependency installation
  - Lambda package creation
  - Deployment to AWS Lambda
  - Integration test execution
```

#### 3. Frontend Application CI/CD

```yaml
Trigger: Changes in frontend/**
Steps:
  - Node.js environment setup
  - npm install dependencies
  - Angular lint (ng lint)
  - Unit tests (ng test)
  - Production build (ng build --prod)
  - Build artifact upload
  - Deployment to S3/Amplify
```

#### 4. Model Training Pipeline

```yaml
Trigger: Manual, Scheduled, Drift detection
Steps:
  - Data quality validation
  - Environment setup
  - Model training execution
  - Evaluation against thresholds
  - Model artifact upload to S3
  - SageMaker model registration
  - Deployment approval gate
```

### Continuous Integration Checks

**Code Quality:**

- Linting: Python (flake8, black), TypeScript (ESLint, Prettier)
- Type checking: mypy for Python, TypeScript compiler
- Code coverage minimum: 80% for critical components
- Documentation completeness validation

**Security Scanning:**

- Dependency vulnerability scanning (npm audit, safety)
- Infrastructure security (checkov for Terraform)
- Secrets detection (GitGuardian)
- IAM policy least-privilege validation

**Testing Strategy:**

- **Unit Tests:** Individual function/component testing
- **Integration Tests:** API Gateway → Lambda → SageMaker flow
- **End-to-End Tests:** Frontend → Backend → Model inference
- **Performance Tests:** Load testing for Lambda/SageMaker endpoints

### Infrastructure as Code

**Terraform Modules:**

1. **Lambda Module** (`lambda.tf`)
   - Function definition with Python 3.11 runtime
   - IAM role with least-privilege policies
   - CloudWatch log group configuration
   - Environment variable management

2. **API Gateway Module** (`api_gateway.tf`)
   - REST API definition
   - CORS configuration
   - Lambda integration
   - Deployment stages (dev, prod)
   - Throttling and rate limiting

3. **Amplify Module** (`amplify.tf`)
   - Frontend hosting configuration
   - GitHub repository integration
   - Build settings for Angular
   - Custom domain support
   - SSL/TLS certificate management

4. **Main Configuration** (`main.tf`)
   - AWS provider configuration
   - S3 bucket for Athena results
   - IAM roles and policies
   - CloudWatch resources
   - Resource tagging for cost allocation

**State Management:**

- Terraform state stored in S3 with versioning
- State locking via DynamoDB
- Separate state files per environment
- Encryption at rest enabled

---

## System Architecture

### Component Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                            │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Angular Web Application (AWS Amplify / S3)            │    │
│  │  - Dashboard UI                                         │    │
│  │  - Anomaly Visualization (Chart.js)                    │    │
│  │  - Real-time Statistics                                │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ↓ HTTPS
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway Layer                           │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  AWS API Gateway (REST API)                            │    │
│  │  - POST /detect-anomalies                              │    │
│  │  - CORS configuration                                  │    │
│  │  - Request throttling                                  │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Compute Layer                               │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  AWS Lambda (Python 3.11)                              │    │
│  │  - Query Athena for sample records                     │    │
│  │  - Feature preparation                                 │    │
│  │  - Invoke SageMaker endpoint                           │    │
│  │  - Result aggregation                                  │    │
│  └────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
             ↓ SQL Query          ↓ Inference Request
┌──────────────────────┐    ┌──────────────────────────────┐
│   Data Lake Layer    │    │   ML Inference Layer         │
│  ┌─────────────┐    │    │  ┌─────────────────────┐    │
│  │ AWS Athena  │    │    │  │ SageMaker Endpoint  │    │
│  │  - SQL      │    │    │  │  - Isolation Forest │    │
│  │  - Catalog  │    │    │  │  - ml.m5.large      │    │
│  └─────────────┘    │    │  │  - Data Capture     │    │
│         ↓           │    │  └─────────────────────┘    │
│  ┌─────────────┐    │    │           ↓                  │
│  │  S3 Buckets │    │    │  ┌─────────────────────┐    │
│  │  - Raw      │    │    │  │ Model Artifacts (S3)│    │
│  │  - Curated  │    │    │  │  - model.tar.gz     │    │
│  │  - Features │    │    │  │  - preprocessor     │    │
│  │  - Preds    │    │    │  │  - metadata.json    │    │
│  └─────────────┘    │    │  └─────────────────────┘    │
└──────────────────────┘    └──────────────────────────────┘
                  ↓                      ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Monitoring Layer                              │
│  ┌────────────────────┐  ┌──────────────────────────────┐      │
│  │  CloudWatch Logs   │  │  SageMaker Model Monitor     │      │
│  │  - Lambda logs     │  │  - Data quality monitoring   │      │
│  │  - API Gateway     │  │  - Feature drift detection   │      │
│  │  - SageMaker logs  │  │  - Anomaly rate tracking     │      │
│  └────────────────────┘  └──────────────────────────────┘      │
│  ┌────────────────────┐  ┌──────────────────────────────┐      │
│  │  CloudWatch Alarms │  │  Model Registry              │      │
│  │  - Drift alerts    │  │  - Version management        │      │
│  │  - Error rates     │  │  - Approval workflow         │      │
│  │  - Latency         │  │  - Model lineage             │      │
│  └────────────────────┘  └──────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **User Interaction:** User accesses Angular dashboard hosted on AWS Amplify/S3
2. **Anomaly Detection Request:** Frontend sends POST request to API Gateway with record count
3. **Lambda Processing:**
   - Receives request and queries Athena for random CMS payment records
   - Prepares features matching training pipeline
   - Invokes SageMaker endpoint with prepared data
4. **Model Inference:** SageMaker returns anomaly scores and predictions
5. **Response Processing:** Lambda aggregates results with statistics and visualizations
6. **Real-time Display:** Frontend renders charts, tables, and metrics
7. **Monitoring:** All operations logged to CloudWatch, data captured for drift detection

### Security Architecture

**Network Security:**

- HTTPS/TLS encryption for all API communications
- API Gateway with AWS IAM authentication
- VPC optional for enhanced endpoint isolation
- Security groups restricting SageMaker endpoint access

**Identity and Access Management:**

- Separate IAM roles for Lambda, SageMaker, Athena
- Least-privilege policies (read/write scoped to specific S3 prefixes)
- No hardcoded credentials (environment variables, AWS Secrets Manager)
- Resource-based policies for cross-service access

**Data Security:**

- S3 bucket encryption at rest (AES-256)
- Athena query results encrypted
- SageMaker data capture encrypted
- CloudWatch logs encrypted with KMS

**Application Security:**

- CORS restricted to frontend domain
- API rate limiting and throttling
- Input validation on Lambda handler
- No PII in logs or monitoring data

---

## Getting Started

### Prerequisites

- AWS Account with appropriate permissions
- Terraform (v1.6+)
- Node.js (v18+) and npm
- Python (v3.11+)
- AWS CLI configured with valid credentials

### Quick Deployment

1. **Clone Repository**

   ```bash
   git clone <repository-url>
   cd aai540_3proj
   ```

2. **Configure Environment**

   ```bash
   cp terraform/terraform.tfvars.example terraform/terraform.tfvars
   # Edit terraform.tfvars with your AWS settings
   ```

3. **Deploy Infrastructure**

   ```bash
   cd terraform
   terraform init
   terraform plan
   terraform apply
   ```

4. **Access Application**
   ```bash
   terraform output amplify_app_url
   ```

### Deployment Options

**Option 1: Automated CI/CD (Recommended)**

- Push to `main` branch triggers automated deployment via GitHub Actions
- Includes linting, testing, and security scanning
- Automatic rollback on failure

**Option 2: Manual Scripts**

- Linux/Mac: `./scripts/deploy.sh`
- Windows: `.\scripts\deploy.ps1`

**Option 3: Terraform Direct**

- Follow the comprehensive guide in [DEPLOYMENT.md](DEPLOYMENT.md)

### Project Structure

```
aai540_3proj/
├── frontend/                  # Angular web application
│   ├── src/app/components/   # Dashboard, charts, tables
│   ├── src/app/services/     # API service layer
│   └── src/environments/     # Environment configs
├── lambda/                    # AWS Lambda functions
│   └── anomaly_detection/    # Anomaly detection API
├── terraform/                 # Infrastructure as Code
│   ├── main.tf               # AWS provider, S3, IAM
│   ├── lambda.tf             # Lambda function definition
│   ├── api_gateway.tf        # REST API configuration
│   └── amplify.tf            # Frontend hosting
├── .github/workflows/        # CI/CD pipelines
├── notebooks/                # Jupyter notebooks with full analysis
├── config/                   # Configuration files
├── data/                     # Sample datasets
└── utils/                    # Utility modules
```

### Application Features

**Dashboard Capabilities:**

- Trigger anomaly detection on sample records from Athena
- Interactive visualizations: pie charts, bar charts, line plots, scatter plots
- Sortable anomaly table with detailed record information
- Real-time statistics and metrics
- Modern, responsive UI with Angular Material and Chart.js

**Backend API:**

- Serverless Lambda function (Python 3.11)
- Athena query execution for data retrieval
- Feature preparation matching training pipeline
- SageMaker endpoint invocation
- Structured JSON response with aggregated results

---

## Monitoring and Cost Optimization

### Observability

- **CloudWatch Logs:** Lambda invocations, API Gateway requests, SageMaker inference
- **CloudWatch Metrics:** Latency, error rates, invocation counts
- **API Gateway Metrics:** Request count, 4XX/5XX errors
- **Amplify Console:** Build logs and deployment status
- **SageMaker Model Monitor:** Data drift, feature drift, anomaly rate drift

### Cost Management

The architecture is optimized for AWS credit constraints:

- **Lambda:** Pay-per-execution (no idle costs)
- **API Gateway:** Pay-per-million calls
- **Amplify:** Free tier available, minimal hosting costs
- **Athena:** Pay per TB scanned (partitioned data reduces costs)
- **SageMaker:** ml.m5.large instance, can be stopped when not in use
- **S3:** Lifecycle policies for automatic data cleanup

**Estimated monthly cost for moderate usage:** $10-50 USD

---

## Development Workflow

### Local Development

**Frontend:**

```bash
cd frontend
npm install
npm start
# Navigate to http://localhost:4200
```

**Lambda Testing:**

```bash
cd lambda/anomaly_detection
pip install -r requirements.txt
python -m pytest tests/  # Run unit tests
```

**Terraform Validation:**

```bash
cd terraform
terraform init
terraform validate
terraform plan
```

### Testing Strategy

- **Unit Tests:** Component and function-level testing
- **Integration Tests:** API Gateway → Lambda → SageMaker flow
- **End-to-End Tests:** Full user workflow validation
- **Model Tests:** Prediction consistency, score distribution validation

---

## Future Enhancements

1. **Data Quality Improvements**
   - Strict validation for state and payment category fields
   - Mapping to approved category lists
   - Separate "data quality anomaly" detection stream

2. **Enhanced Features**
   - Entity-level behavioral scoring (recipient + payer history)
   - Rolling aggregations and z-scores within peer groups
   - Relationship count features and sudden-change indicators

3. **Model Improvements**
   - Local Outlier Factor (LOF) for density-based detection
   - Semi-supervised learning from reviewer feedback
   - SHAP explanations for high-risk predictions

4. **Operational Enhancements**
   - Multi-level scoring (recipient, company, specialty benchmarks)
   - Extend to Research Payments and Ownership datasets
   - Automated data quality monitoring rules

---

## Security, Privacy, and Ethics

### Data Privacy

- **Dataset:** CMS Open Payments (publicly available)
- **PII Present:** Yes (provider identity/location)
- **PHI Present:** No
- **Mitigation:**
  - Encrypt storage (S3, SageMaker)
  - Restrict access with IAM least-privilege policies
  - Exclude names and identifiers from model features
  - Results framed as "review prioritization," not fraud determination

### Bias Considerations

- Payment patterns vary legitimately across specialties, regions, and institution types
- Peer-group comparisons reduce bias from inherent specialty/geography differences
- Subgroup monitoring tracks anomaly rates by recipient type, specialty, and state
- Regular audits ensure data quality issues don't disproportionately flag certain groups

### Ethical Framework

- **Intended Use:** Prioritize compliance reviews of unusual payment patterns
- **Not Intended:** Definitive fraud claims, legal determinations, automated enforcement
- **Transparency:** Align with CMS guidance on Open Payments interpretation
- **Accountability:** Model cards document limitations and monitoring requirements

---

## Technologies and Tools

- **Machine Learning:** scikit-learn (Isolation Forest), TensorFlow (Autoencoder), XGBoost
- **Cloud Platform:** AWS (S3, Athena, Lambda, API Gateway, SageMaker, Amplify, CloudWatch)
- **Infrastructure:** Terraform, GitHub Actions
- **Frontend:** Angular 17, TypeScript, Angular Material, Chart.js
- **Backend:** Python 3.11, Boto3
- **Data Processing:** Pandas, NumPy, Parquet
- **Monitoring:** CloudWatch, SageMaker Model Monitor

---

## References

Centers for Medicare & Medicaid Services. (2025a). _Open Payments: Program overview and data updates (Program Year 2024 publication)_. Open Payments. https://openpaymentsdata.cms.gov/datasets/download

Centers for Medicare & Medicaid Services. (2025b). _Open Payments data dictionary / methodology documentation for public use files_. Open Payments. https://openpaymentsdata.cms.gov/dataset/e6b17c6a-2534-4207-a4a1-6746a14911ff#data-dictionary

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (n.d.). IsolationForest. In scikit-learn documentation. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

Hinton, G. E., & Salakhutdinov, R. R. (2006). Reducing the dimensionality of data with neural networks. _Science, 313_(5786), 504–507. https://doi.org/10.1126/science.1127647

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. In _Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining_ (pp. 785-794). https://arxiv.org/abs/1603.02754

Amazon Web Services. (n.d.). _Model registration and deployment with Amazon SageMaker Model Registry_. https://docs.aws.amazon.com/sagemaker/latest/dg/model-registry.html

Amazon Web Services. (n.d.). _Deploy models for real-time inference—Amazon SageMaker AI_. https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints-deploy-models.html

---

## Additional Documentation

- **[DEPLOYMENT.md](DEPLOYMENT.md)** - Comprehensive deployment guide
- **[Frontend README](frontend/README.md)** - Frontend-specific documentation
- **[Lambda README](lambda/anomaly_detection/README.md)** - Lambda function details
- **[Notebook](notebooks/cms_anomaly_detection.ipynb)** - Complete analysis and results

---

## License

This project is developed for educational purposes as part of AAI-540 Machine Learning Operations at University of San Diego.

---

## Contact

For questions or collaboration inquiries, please reach out to the team members or create an issue in the repository.
