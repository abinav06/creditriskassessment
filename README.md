# End-to-End Credit Risk Assessment System

This project demonstrates a complete, end-to-end machine learning workflow for assessing credit risk on the Lending Club loan dataset. It covers the entire lifecycle, from data cleaning and feature engineering to model training, optimization, and deployment as a real-time REST API.

The final system predicts the probability of a loan defaulting and provides a risk assessment via an interactive API endpoint.

**Live API Documentation (once running):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

![SHAP Plot](path/to/your/shap_plot_image.png)
*(This SHAP plot shows the key drivers influencing the model's predictions after data leakage was resolved)*

---

## Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run the Project](#how-to-run-the-project)
- [Model Performance](#model-performance)
- [Future Improvements](#future-improvements)

---

## Project Overview

The goal of this project is to build a reliable credit risk model that can predict the likelihood of a loan applicant defaulting. This project stands out by focusing on building a trustworthy and deployable system, which involved:

1.  **Rigorous Data Cleaning:** Handling missing values and identifying and resolving critical data leakage issues inherent in the dataset.
2.  **Feature Engineering:** Creating new, insightful features from existing data, such as income-to-loan ratios.
3.  **Model Optimization:** Systematically tuning a LightGBM model to achieve the best possible performance.
4.  **Deployment as a Service:** Encapsulating the entire preprocessing and prediction workflow into a single pipeline and deploying it as a REST API.

---

## Key Features

- **End-to-End ML Pipeline:** A complete, runnable workflow from raw data to a deployed API.
- **Robust Model Validation:** Uses 5-fold cross-validation to ensure the model's performance is stable and reliable.
- **Hyperparameter Tuning:** Leverages **Optuna** for efficient and systematic optimization of the LightGBM model.
- **Explainable AI (XAI):** Uses **SHAP** to interpret the model's predictions and understand the key factors driving risk.
- **Production-Ready Artifacts:** The entire data preprocessing and modeling logic is saved as a single, serialized scikit-learn Pipeline using `joblib`.
- **Live API Endpoint:** A **FastAPI** application serves the model, allowing for real-time predictions on new loan application data.

---

## Tech Stack

- **Language:** Python 3.10+
- **Data Manipulation:** pandas, NumPy
- **Machine Learning:** scikit-learn, LightGBM
- **Hyperparameter Optimization:** Optuna
- **Model Interpretation:** SHAP
- **API Development:** FastAPI, Uvicorn
- **Serialization:** joblib

---

## Setup and Installation

### 1. Clone the repository:
```bash
git clone https://github.com/your-username/credit-risk-system.git
cd credit-risk-system
```

### 2. Download the Dataset:
The dataset is not included in this repository due to its large size.
- You can download it from Kaggle: [Lending Club Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club)
- After downloading, unzip the file if necessary, and place the `lending_club_loan.csv` file directly into the **root directory** of this project.

### 3. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```

### 4. Install the required dependencies:
*(A `requirements.txt` file is recommended. To create one, run `pip freeze > requirements.txt` after installing the packages below)*
```bash
pip install pandas scikit-learn lightgbm optuna shap fastapi "uvicorn[standard]"
```

---

## How to Run the Project

Follow these steps in order to replicate the project. Ensure the `lending_club_loan.csv` file is in the root directory before starting.

### Step 1: Find the Best Hyperparameters
This script runs an Optuna study to find the optimal settings for the LightGBM model.
```bash
python -m ml.tune_hyperparams
```
Wait for this to complete. It will print a dictionary of the best parameters found.

### Step 2: Train the Final Model
1.  **Copy** the `Best Parameters` dictionary printed from the previous step.
2.  **Open** the `ml/train.py` file and **paste** the dictionary into the `best_params` variable.
3.  Run the training script to create the `credit_risk_pipeline.joblib` file in the `artifacts/` folder.
    ```bash
    python -m ml.train
    ```

### Step 3: Run the API Server
This command starts the web server, which will load the saved pipeline and listen for requests.
```bash
uvicorn ml.api:app --reload
```
The server will be running at `http://127.0.0.1:8000`.

### Step 4: Test the API
1.  Open your web browser and navigate to the interactive API documentation at `http://127.0.0.1:8000/docs`.
2.  Expand the `POST /predict` endpoint, click **"Try it out"**.
3.  Fill in the request body with data for a loan application (an example is provided in the documentation).
4.  Click **"Execute"** to see the model's live prediction.

---

## Model Performance

- **Final Model:** LightGBM Classifier
- **Validation Strategy:** 5-Fold Stratified Cross-Validation
- **Performance Metric:** Area Under the ROC Curve (AUC)
- **Result:** **`Average CV AUC: 0.71`**

This robust score was achieved after resolving data leakage and performing systematic hyperparameter tuning, indicating a reliable and realistic model performance.

---

## Future Improvements

- **Scale to Full Dataset:** Train the final model on the entire dataset to potentially improve performance and generalization.
- **Advanced Feature Engineering:** Incorporate more complex features, such as interaction terms or target-encoding.
- **Time-Based Validation:** Implement a time-based cross-validation split to better simulate real-world model deployment.
- **Containerization:** Dockerize the FastAPI application to make deployment even more portable and scalable.
- **Automated CI/CD:** Set up a GitHub Actions workflow to automatically test the code and deploy the API on push.