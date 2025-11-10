# AI-workflow-development-week-5
Part 2 — Case Study: Hospital readmission within 30 days (40 points)

1. Problem Scope (5 points)

Problem definition: Predict which patients discharged from the hospital are at high risk of being readmitted within 30 days.

Objectives
	1.	Reduce avoidable 30-day readmissions by enabling targeted post-discharge care.
	2.	Allocate follow-up resources (home visits, phone calls) efficiently to high-risk patients.
	3.	Provide interpretable risk drivers so clinicians can act (e.g., medication adherence, comorbidity management).

Stakeholders
	•	Hospital clinicians and discharge planners (use predictions to plan care).
	•	Patients (benefit from reduced readmission risk and targeted care).
	•	Hospital administration / payers (financial and quality metrics).

⸻

2. Data Strategy (10 points)

Proposed data sources
	1.	Electronic Health Records (EHR): diagnoses (ICD codes), lab results, vital signs, medication lists, comorbidities, previous admissions.
	2.	Administrative & claims data: prior utilization, insurance type, length of stay, discharge disposition.
(Optionally: social determinants of health — housing stability, transportation access — from intake forms or linked public datasets.)

Two ethical concerns
	1.	Patient privacy & data security: EHR data is highly sensitive; must comply with HIPAA (or local equivalents) and minimize re-identification risk.
	2.	Bias and fairness: model could systematically under/over-predict risk for subgroups (by race, language, socioeconomic status), leading to unequal access to interventions.

Preprocessing pipeline (including feature engineering)
	1.	Data ingestion & linkage
	•	Extract EHR records for relevant time window; link by patient ID; de-identify where possible for development.
	2.	Missing-data handling
	•	Use clinically-informed imputations (e.g., carry-forward last observation for vitals), and add missingness indicators for labs not ordered.
	3.	Temporal aggregation & windowing
	•	Aggregate time-series into features over windows (e.g., last 30/90 days): counts of ED visits, avg lab values, slope of creatinine.
	4.	Feature engineering
	•	Comorbidity scores (Charlson index), polypharmacy indicator (# medications), discharge disposition (home vs skilled nursing), length-of-stay, prior admissions count.
	•	Social factors: a binary indicator for unstable housing if present.
	5.	Encoding & scaling
	•	Encode categorical fields (one-hot/target encode), standardize continuous features.
	6.	Train/test split with temporal holdout
	•	Keep latest quarter/year as test set to simulate real deployment and prevent leakage.
	7.	Label creation
	•	Define label = readmission within 30 days (exclude planned readmissions if possible).
	8.	Data quality checks & privacy controls
	•	Remove direct identifiers, log data provenance, restrict access via role-based permissions.

⸻

3. Model Development (10 points)

Model selection & justification
	•	Model: Gradient Boosted Trees (LightGBM/XGBoost) or a logistic regression with engineered features.
	•	Justification: Readmission prediction is tabular, benefits from handling mixed types and missingness. GBT provides strong predictive power and feature importance; logistic regression may be used as a simple, interpretable baseline for clinical acceptance.

Hypothetical confusion matrix & calculations
(We show an example on a test set of 1,000 

4. Deployment (10 points)
   

Steps to integrate the model into the hospital’s system
	1.	Productionize model artifact — export model as a serialised object (e.g., LightGBM model file) and containerize predictor in Docker.
	2.	Feature pipeline productionization — implement a reliable feature extraction service (ETL) that runs on nightly batch or near-real-time from the EHR; ensure transformations in training are identically applied in production (use feature store or same code).
	3.	Model serving — deploy model behind an internal API (REST/gRPC) with authentication, or integrate into existing clinical decision support (CDS) systems. Support batch scoring for daily discharge lists and on-demand scoring at discharge time.
	4.	UI & clinician workflow — surface risk scores and top contributing features in the discharge planner UI; keep explanations short and actionable (e.g., top 3 risk drivers).
	5.	Logging & monitoring — log inputs, predictions, and downstream outcomes (readmissions) for auditing, monitoring, and retraining. Implement alerting for performance degradation.
	6.	Rollout & feedback loop — start with shadow deployment or limited pilot, collect clinician feedback, then phased rollout; have a retraining schedule.

Ensuring compliance with healthcare regulations (e.g., HIPAA)
	•	Data minimization & access control: only store/process PHI necessary for the model; enforce role-based access and audit logs.
	•	Encryption: encrypt data at rest and in transit (TLS, disk encryption).
	•	Business Associate Agreements (BAAs): ensure third-party vendors (cloud providers) have BAAs in place.
	•	De-identification for development: use de-identified or limited datasets for model development when possible.
	•	Documentation & explainability: maintain model documentation, validation reports, and decision explanations for clinical audit and regulatory review.
	•	Incident response & breach notification procedures aligned with HIPAA rules.



5. Optimization (5 points)

One method to address overfitting
	•	Regularization (e.g., L2 for linear models or controlling tree complexity & early stopping for GBTs).
	•	For GBTs: limit tree depth / number of leaves, use learning rate decay, and apply early stopping on a validation set.
	•	This reduces model variance and improves generalization.

# Sample Data + Model Training Example
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# -----------------------------
# 1. Create Sample (Synthetic) Data
# -----------------------------
np.random.seed(42)

n = 500   # number of patients

data = pd.DataFrame({
    "age": np.random.randint(20, 90, n),
    "num_prior_admissions": np.random.poisson(1.5, n),
    "length_of_stay": np.random.randint(1, 15, n),
    "chronic_conditions": np.random.randint(0, 5, n),
    "med_count": np.random.randint(1, 12, n),
    "has_followup_appointment": np.random.choice([0,1], n),
})

# True pattern: higher conditions + long stay + no follow-up increases readmission risk
data["readmitted_30_days"] = (
    (data["chronic_conditions"] * 0.4) +
    (data["length_of_stay"] * 0.1) +
    (1 - data["has_followup_appointment"]) * 0.8 +
    np.random.normal(0, 0.5, n)
)

# Convert to binary label ( > median risk = high risk )
threshold = data["readmitted_30_days"].median()
data["readmitted_30_days"] = (data["readmitted_30_days"] > threshold).astype(int)

# -----------------------------
# 2. Train/Test Split
# -----------------------------
X = data.drop("readmitted_30_days", axis=1)
y = data["readmitted_30_days"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# 3. Train a Model
# -----------------------------
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# -----------------------------
# 4. Evaluate the Model
# -----------------------------
y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nPrecision:", round(precision, 3))
print("Recall:", round(recall, 3))

Part 3: Critical Thinking (20 points)

⸻

1. Ethics & Bias (10 points)
How biased training data may affect outcomes:
If the historical hospital data contains bias (e.g., certain groups received different levels of care due to socioeconomic status, language barriers, age, or race), the model may learn and reinforce those inequalities.
For example, if patients from low-income backgrounds historically had higher readmission rates due to reduced access to follow-up care, the model may label all similar patients as “high-risk”, even when their clinical condition is stable.
This can lead to:
	•	Unequal resource allocation (some groups receive more interventions, others less).
	•	Unfair patient treatment and worsened trust in healthcare systems.
	•	Amplification of existing health disparities over time.

One strategy to mitigate this bias:
	•	Perform fairness auditing and reweighting.
Analyze model performance across demographic groups and adjust training weights or thresholds so that the model performs equally well across subgroups.
For example, ensure recall and precision are balanced across age, gender, and socioeconomic groups.

⸻

2. Trade-offs (10 points)
Interpretability vs Accuracy in Healthcare:
	•	In healthcare, interpretability is critical because clinicians need to understand why a prediction is made before acting on it.
	•	Highly complex models (e.g., deep neural networks) may provide higher accuracy, but they are often difficult to explain.
	•	Simpler models (e.g., logistic regression, decision trees) are easier to interpret but may have slightly lower performance.
	•	Therefore, the trade-off is between better predictive performance versus clinically acceptable transparency.
Most hospitals prefer interpretable models because treatment decisions must be justifiable and defensible.
Impact of Limited Computational Resources on Model Choice:
	•	If the hospital has limited computing capacity, they may need to choose:
	•	Simpler models (e.g., logistic regression, random forest with small depth)
	•	Models that run quickly in real-time on standard hospital hardware
	•	Complex models like large neural networks or large gradient-boosted ensembles may be too slow or expensive to deploy.
	•	Therefore, the hospital may prioritize efficiency and reliability over small gains in accuracy.



