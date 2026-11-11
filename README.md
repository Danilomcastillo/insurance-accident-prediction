# 🏔️ Mountain Insurance Accident Prediction

An end-to-end Machine Learning pipeline predicting seasonal accident risk for the **Madrid Mountain Federation** (121k+ records). Built as a final capstone project, optimized for leak-free cross-validation and risk factor interpretability.

---

## 🎯 Business Problem & Key Challenges
Insurance providers and mountain federations need to identify high-risk member profiles to optimize prevention strategies and policy pricing.

* **Class Imbalance:** Extreme target asymmetry (~4.7% positive cases).
* **Data Leakage Risk:** High risk of data leakage both in temporal features and across repeated member IDs over multiple seasons.
* **Administrative Noise:** Structural noise from inactive members renewing licenses solely for non-sports reasons.

---

## 💡 Key Engineering & Methodological Highlights

1. **Leakage Prevention:** 
   - Identified and dropped `numeroaccidentesaño` (correlation = 0.95 with target, capturing future accidents from the same evaluation year).
   - Applied **`GroupShuffleSplit` & `GroupKFold`** grouped by member ID (`nsocio`) to prevent data leakage of the same individual across train/test sets.

2. **Feature Engineering:**
   - **Ordinal Encoding:** Encoded 7 license risk levels based on coverage intensity.
   - **Skew Reduction:** Log-transformed (`log1p`) long-tailed historical accidents (`accidenteshistoricos`).
   - **Interaction Features:** Created `grupo_persona` combining age and gender/category demographic buckets.

3. **Model Selection & Trade-offs:**
   - Evaluated Logistic Regression, XGBoost, and Random Forest using `GridSearchCV`.
   - **PR-AUC (0.117)** was leveraged alongside **AUC-ROC (0.69)** to accurately evaluate performance under severe class imbalance (more than 2x over random baseline).

---

## 📊 Model Performance & Insights

| Model | AUC-ROC | Notes |
| :--- | :---: | :--- |
| **XGBoost (Tuned)** | **0.6934** | Best overall performance handling non-linear interactions |
| **Logistic Regression** | **0.6808** | Highest interpretability via Odds Ratios |
| **Random Forest** | 0.6011 | Baseline tree ensemble |

### 🔍 Risk Factor Interpretability (Odds Ratios)
* **Historical Accidents ($\text{OR} = 5.64$):** Each past accident multiplies the probability of a new accident by $\sim5.6\times$.
* **License Coverage ($\text{OR} = 1.57$):** Higher coverage licenses correlate directly with higher risk activities.
* **Age ($\text{OR} = 0.75$):** Older members show lower individual accident rates.

---

## 🛠️ Tech Stack
* **Language:** Python 3.12
* **Data & ML:** Pandas, NumPy, Scikit-learn, XGBoost, SciPy
* **Visualization:** Seaborn, Matplotlib

---

## 📁 Repository Structure
```text
insurance-accident-prediction/
│
├── data/
│   ├── raw/             # Excluded from git (Confidential PII data)
│   └── processed/       # Sanitized final dataset for modeling
│
├── notebooks/           # Exploratory Data Analysis & Model Experiments
├── src/                 # Modular Python scripts
├── presentation/        # Final executive presentation (PDF)
├── README.md
└── requirements.t
