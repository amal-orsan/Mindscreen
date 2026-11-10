<div align="center">

# 🧠 MindScreen

### Depression assessment & clinical decision-support dashboard

A Streamlit app that combines **PHQ-based clinical rules** with a **machine-learning pipeline** to screen depression severity and subtype, explain the contributing factors, and suggest evidence-based next steps.

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-Database-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

> ⚠️ **Disclaimer:** MindScreen is a graduation / educational project. It is a **decision-support prototype, not a medical device**, and does **not** replace evaluation by a qualified mental health professional. Do not use it to diagnose or treat anyone.
>
> **If you or someone you know is in crisis:** call or text **988** (US Suicide & Crisis Lifeline), text **HOME** to **741741**, or find local help at [findahelpline.com](https://findahelpline.com).

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [How It Works](#-how-it-works)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Dataset](#-dataset)
- [Model Experiments & Results](#-model-experiments--results)
- [Database Schema](#-database-schema)
- [Limitations](#-limitations)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

---

## 🔎 Overview

MindScreen lets a clinician (or a student exploring the project) enter a patient's demographic information, lifestyle indicators, a PHQ depression score, and a symptom/risk checklist. The app then returns:

- a **depression category** with a confidence score,
- the **top-3 differential diagnoses**,
- **risk flags** (e.g. suicide / self-harm history, sleep deficit, worsening trend),
- a plain-language **clinical-factor explanation**, and
- **treatment recommendations** matched to the result.

Every assessment is stored in a local SQLite database, so results can be browsed, filtered, exported, and summarised in a statistics dashboard.

<!--
📸 Add screenshots here once you have them, e.g.:
![Assessment tab](docs/assessment.png)
![Records tab](docs/records.png)
![Statistics tab](docs/statistics.png)
-->

## ✨ Features

| Tab | What it does |
|---|---|
| 🔍 **Assessment** | Input form (demographics, PHQ score, sleep, social media, anxiety, eating habits, symptom checklist, perinatal status) → prediction, confidence bar, risk flags, differential diagnoses, clinical factors, recommendations |
| 📋 **Patient Records** | Browse saved assessments; search by diagnosis, filter by severity, sort by date / PHQ / risk; **export to CSV** |
| 📊 **Statistics** | KPIs (total, high-risk, average PHQ, average confidence, last 7 days) plus severity and diagnosis distributions |

**Also included**

- 🚨 Automatic **crisis-resources panel** for severe results or any suicide/self-harm history
- 🤰 Dedicated **perinatal / postpartum** pathway for female patients
- 🗄️ Normalised SQLite storage (patients, assessments, risk flags, recommendations)
- 🧪 Reproducible ML experiments across 9 algorithms with stratified 5-fold cross-validation

## ⚙️ How It Works

```
 Streamlit form ──► encode inputs ──► rule-based clinical logic ──┬──► label + confidence + top-3
                                                                  │
                                       ML model (model.pkl) ◄─────┘  (used for cases the rules don't settle)
                                                                  │
                                                                  ▼
                              risk flags · clinical factors · recommendations ──► SQLite
```

1. **Encoding** – form inputs are mapped to the same numeric scheme used to train the model (education, employment, coping method, overeating level, meals/day, etc.).
2. **Clinical rules** – the PHQ score drives the primary decision (0–4 none, 5–9 mild, 10–14 moderate, 15–19 moderately-severe, 20+ severe), with refinements for situational depression, peripartum depression, and severe depression with prominent anxiety.
3. **ML fallback** – the trained scikit-learn pipeline in `model.pkl` handles cases the rules deliberately leave open (for example, mild PHQ with a self-harm or suicide history). If the model file is missing, the app falls back to a PHQ-only **demo mode** and says so.
4. **Explanation & storage** – the app builds a ranked list of clinical factors and recommendations, then persists everything to `mindscreen.db`.

## 🗂 Project Structure

```
Mindscreen/
├── app.py                      # Streamlit application (UI + inference logic)
├── database.py                 # SQLite layer: schema, inserts, queries, stats
├── model.pkl                   # Trained ML pipeline used by the app
├── mindscreen.db               # Local SQLite database (created at runtime)
├── data/
│   └── Mental Health Classification.csv
├── helpers/
│   ├── imports.py              # Shared imports for experiments
│   ├── read_data.py            # Loads data, defines X / y, drops leakage column
│   ├── read_data.ipynb
│   └── info_about_data.ipynb   # EDA: distributions & data-quality checks
├── preprocessing/
│   ├── preprocessor.py         # ColumnTransformer (one-hot + scaling)
│   └── Preprocessing_.ipynb
├── split_data/
│   ├── cross_val_split.py      # Stratified 5-fold CV, scoring, pipeline builder
│   └── cross_val_split.ipynb
└── models/                     # One notebook per algorithm
    ├── logistic_regression.ipynb   ├── svm.ipynb
    ├── knn.ipynb                   ├── naive_bayes.ipynb
    ├── decision_tree.ipynb         ├── random_forest.ipynb
    ├── xgboost.ipynb               ├── ligthgbm.ipynb
    └── transfer.ipynb              # TabPFN (transfer learning)
```

## 🚀 Getting Started

### Prerequisites

- Python **3.9+**
- `pip` (or conda)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/amal-orsan/Mindscreen.git
cd Mindscreen

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install streamlit pandas numpy joblib scikit-learn imbalanced-learn xgboost lightgbm
```

> For the experiment notebooks you will also need `jupyter`, `matplotlib`, and `seaborn`, plus `tabpfn` for `models/transfer.ipynb`.
>
> **Tip:** `model.pkl` was serialised with specific library versions. If loading fails, install the versions it was trained with (or re-train and re-save the model).

### Run the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually <http://localhost:8501>). The SQLite database is created automatically on first run.

### Reproduce the experiments

```bash
cd models
jupyter notebook          # open any model notebook and run all cells
```

Each notebook imports the shared helpers, preprocessing, and cross-validation setup, so every algorithm is evaluated under identical conditions.

## 📊 Dataset

- **File:** `data/Mental Health Classification.csv`
- **Size:** 1,998 rows × 21 columns, no missing values, no duplicate rows
- **Target:** `Depression_Type` (12 classes, 0–11; strongly imbalanced)
- **Feature groups**
  - *Demographic:* gender, age, education level, employment status
  - *Behavioural:* social-media hours, social media while eating, meals per day, overeating level, sleep hours, online depression searches, mental-health support, coping methods
  - *Psychological:* low energy, low self-esteem, worsening depression, nervousness, depression score, self-harm, suicide attempts
- **Leakage handling:** the `Symptoms` column is dropped before training (`LEAKAGE_COLS` in `helpers/read_data.py`).

## 🧪 Model Experiments & Results

All models share one pipeline: **preprocessing → SMOTETomek (class balancing) → classifier**, evaluated with **stratified 5-fold cross-validation** (`random_state=42`).

- *Preprocessing:* one-hot encoding for `Employment_Status` and `Coping_Methods`; standard scaling for `Age`, `Sleep_Hours`, `SocialMedia_Hours`, `Depression_Score`; remaining columns passed through.

| Model | Val. Accuracy | Val. F1 (weighted) | Val. F1 (macro) |
|---|:---:|:---:|:---:|
| Naive Bayes | 0.5185 | 0.5213 | 0.7772 |
| Decision Tree (depth 10) | 0.7132 | 0.6932 | 0.8857 |
| Logistic Regression | 0.9229 | 0.9227 | 0.9722 |
| SVM (linear) | 0.9480 | 0.9515 | 0.9535 |
| KNN (k=5) | 0.9640 | 0.9646 | 0.9664 |
| Random Forest (100 trees, depth 10) | 0.9955 | 0.9955 | 0.9986 |
| XGBoost | 1.0000 | 1.0000 | 1.0000 |
| LightGBM | 1.0000 | 1.0000 | 1.0000 |
| TabPFN (transfer learning) | see `models/transfer.ipynb` | | |

> 🔬 **A note on the near-perfect scores.** Perfect cross-validation results from XGBoost and LightGBM on a ~2,000-row tabular dataset usually mean the labels are almost fully determined by a few features (here, most likely `Depression_Score`) or that the data is synthetic / highly structured. These numbers should be read as *"the pattern is easy to learn from this dataset"*, not as proof of real-world clinical accuracy. External validation on independent clinical data would be required before drawing any such conclusion.

## 🗄 Database Schema

SQLite (WAL mode, foreign keys enabled), defined in `database.py`:

| Table | Purpose |
|---|---|
| `patients` | Demographics, clinical scores, and behavioural / risk flags for each submission |
| `assessments` | Prediction output (class, label, severity, confidence, top-3 differential as JSON, clinician notes) linked to a patient |
| `risk_flags` | Per-assessment risk tags (normalised) |
| `recommendations` | Per-assessment ordered treatment recommendations (normalised) |

Indexes exist on patient, date, severity, and predicted class for fast filtering.

## ⚠️ Limitations

- **Not clinically validated.** Outputs are probabilistic estimates for decision support only.
- **Rules take priority over the model.** For most PHQ ranges, the rule-based logic determines the result; the ML model is only consulted for cases the rules leave open.
- **Label coverage.** The dataset has 12 depression types, while the app's label set covers 10 (0–9).
- **Scale naming.** The 0–27 depression score is labelled "PHQ-10" in the interface; the standard instrument with a 0–27 range is the **PHQ-9**.
- **Small, imbalanced dataset** with a limited age range; results may not generalise to other populations.
- **Sensitive data.** The app stores patient-style records locally in plain SQLite with no authentication. Do not enter real patient data in a public or shared deployment.

## 🛣 Roadmap

- [ ] Add `requirements.txt` with pinned versions
- [ ] Replace the rule layer with a calibrated, validated model (or hybrid) and report held-out test metrics
- [ ] Feature-importance / SHAP explanations
- [ ] User authentication and encrypted storage
- [ ] Docker image and one-click deployment (Streamlit Community Cloud)
- [ ] Multilingual interface (Arabic / English)
- [ ] Unit tests for encoding, rules, and database helpers

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit your changes: `git commit -m "Add my feature"`
4. Push the branch: `git push origin feature/my-feature`
5. Open a Pull Request

## 📄 License

Distributed under the **MIT License**. Add a `LICENSE` file to the repository root to make this official.

## 👩‍💻 Author

**Amal Orsan** — [@amal-orsan](https://github.com/amal-orsan)

Built as a graduation project. If you find it useful, consider giving it a ⭐!
