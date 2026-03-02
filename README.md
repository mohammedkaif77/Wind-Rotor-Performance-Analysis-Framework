
# 🌬️ Wind Rotor Performance Analysis Framework

An end-to-end Machine Learning engineering project for analyzing and predicting wind turbine rotor performance using synthetic wind datasets and automated CI/CD pipelines.

---

## 🚀 Project Overview

The **Wind Rotor Performance Analysis Framework** simulates wind behavior, trains predictive machine learning models, and validates performance automatically through a structured CI/CD workflow.

This project demonstrates:

- Synthetic wind dataset generation
- Physics-inspired feature modeling
- Machine Learning regression training
- Performance evaluation (RMSE, R²)
- Automated validation using GitHub Actions
- Artifact generation and reporting

It follows clean project structuring and foundational MLOps principles.

---

## 🏗️ Architecture

Developer → GitHub Repository → GitHub Actions → Artifacts & Reports

### 🔄 CI/CD Pipeline Stages

On every push to the `main` branch:

1. Setup Python environment  
2. Install dependencies  
3. Generate synthetic dataset  
4. Train ML model  
5. Evaluate performance  
6. Save model artifact  
7. Upload reports  

This ensures reproducibility and continuous validation.

---

## 📂 Project Structure

```

Wind-Rotor-Performance-Analysis-Framework/
│
├── .github/
│   └── workflows/
│       └── main.yml          # CI/CD configuration
│
├── data/                     # Generated wind datasets
├── reports/                  # Model outputs & evaluation reports
├── src/                      # Core ML and data logic
├── tests/                    # Unit tests
│
├── main.py                   # Pipeline entry point
├── requirements.txt          # Project dependencies
└── README.md

````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/mohammedkaif77/Wind-Rotor-Performance-Analysis-Framework.git
cd Wind-Rotor-Performance-Analysis-Framework
````

Create virtual environment:

```bash
python -m venv venv
source venv/bin/activate     # Linux/Mac
venv\Scripts\activate        # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Pipeline

```bash
python main.py
```

This will:

* Generate synthetic wind dataset
* Train regression model
* Evaluate performance
* Save trained model
* Generate reports

---

## 📊 Model Evaluation

Metrics used:

* Root Mean Squared Error (RMSE)
* R² Score

These metrics validate prediction accuracy for wind power estimation.

---

## 🛠 Tech Stack

* Python
* NumPy
* Pandas
* Scikit-learn
* GitHub Actions (CI/CD)

---

## 🎯 Engineering Highlights

* Modular project structure
* Automated ML validation via CI/CD
* Reproducible training pipeline
* Artifact packaging and upload
* Separation of data, logic, and testing

---

## 🚀 Future Improvements

* FastAPI inference API
* Docker containerization
* Cloud deployment (AWS / GCP)
* Model versioning
* Monitoring integration

---

## 👨‍💻 Author

Mohammed Kaif
Aspiring Machine Learning Engineer
Building reproducible ML systems with CI/CD practices

