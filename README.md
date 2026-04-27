# Secom Failure Classification (End-to-End ML + MLOps)
## Overview
This project focuses on building a robust machine learning pipeline to detect manufacturing failures using the SECOM dataset (high-dimensional, highly imbalanced data).

The workflow covers:
- Data preprocessing & feature reduction
- Model experimentation & selection
- MLflow-based experiment tracking
- Threshold tuning for imbalanced classification
- API deployment using FastAPI
- Containerization with Docker
- Cloud deployment on AWS EC2

## Project Structure
```
├── app/     
│   ├── main.py                # FastAPI Application
│   ├── config.json            # contains selected features and threshold
│
├── model/                     # Saved Model
│   └── exported_model/
├── Secom_Notebook.ipynb       # Notebook file
├── Dockerfile                 # File for Dockerization
├── requirements.txt           # requirements for application
├── .dockerignore
├── .gitignore
└── .github/
    └── workflows/
        └── ci.yml
```
## Results
**Top K- experimentation aganist PR-AUC Scores**

![K-values](feature_performance.png)

*K=100 has the highest PR-AUC score of 0.23*

**Different Model Performances against PR-AUC Scores on selected top-100 features**
![Model-selection](model_selection.png)

*Selected **Random Forest** as best performing model with PR-AUC of 0.26*

**Hyperparameter Tuning of the best model (RandomForest) with selected Top-100 features**

![parameter_tuning](parameter_tuning.png)

Best Parameters found: {n_estimators: 100, max_depth: 10, min_split: 5, min_leaf: 1} *resulted in pr_auc score: 0.331 and roc-auc score: 0.782*

**26.92% improvement from baseline in pr_auc acheieved by hyperparamter tuning**


Predicted Probabilities on test data and evaluated **Precision**, **Recall**, **F1-Score** at multiple thresholds and optimal threshold is selected based on max **F1-Score** for balanced performance.

**Untuned Performance**

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Fail)** | 0.93 | 1.00 | 0.97 | 293 |
| **1 (Pass)** | 0.00 | 0.00 | 0.00 | 21 |
| **Accuracy** | | | **0.93** | 314 |

![threshold_tuning](threshold_tuning.png)

**Tuned Performance** *@0.217*

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **0 (Pass)** | 0.96 | 0.97 | 0.97 | 293 |
| **1 (Fail)** | 0.56 | 0.48 | 0.51 | 21 |
| **Accuracy** | | | **0.94** | 314 |

**Improved Minority Class F1 from 0 → 0.51**\
This model is saved in local directory using `mlflow.sklearn.save_model(model,"exported_model")`

## Dataset Details
* Rows: 1567
* Columns: 592
* Problem Type: Binary Classification (Highly Imbalanced) (0:1463, 1:104) with target mapping ({-1:0,1:1})

## Data Preprocessing Pipeline
**1. Missng Value Analysis :** Computed null percentage per columns and selected a cutoff of 60 % through histogram, dropping columns having >60% of null values.\
**2. Feature Reduction :** Applied Variance Threshold reducing 566 → 301 columns.\
**3. Imputation :** Used Median to impute Missing Values.\
**4. Stratified Train Test Split :** to preserve class distribution in both train and test datasets.

## Feature Selection Strategy
* Used **SelectKBest(f_classif)**
* Evaluated multiple *k* values: [50,75,100,125,150,175,200]
* Tracked experiments using **MLFlow**
* Metric used: **PR-AUC** for comparision across k values on different folds (n_splits=5) aggregated PR-AUC across folds for different k values on **validation data** subsetted from training data with preserved class balance.
**Selected Top k(100)** features based on the best aggregated **PR-AUC** across splits for stability.

## Model Training & Comparision
Trained multiple models with selected top 100 features evaluated using PR-AUC
* RandomForest
* XGBoost
* LightGBM
* Logistic Regression (with scaled data)

Handling Imbalance
* `class_weight="balanced"` (where applicable)
* LightGBM → `scale_pos_weight`\
**Best Model: RandomForest, PR-AUC: 0.26** (outperformed all other models)

## Hyperparameter Tuning on test data
* Tuned:
  * `n_estimators`
  * `max_depth`
  * `min_samples_split`
  * `min_samples_leaf`
* Logged every run in MLFlow
* Selected best configuration based on PR-AUC

## Configuration Management 
Saved in (config.json):
* Selected features (Top 100)
* Optimal threshold

## API Deployment
Built inference API using **FastAPI**
Features:
* Loads trained model
* Arranges features in the training order of the model using `config.json`
* Raises error on missing features
* Used Tuned threshold
* Returns classification output and probability of pass or fail

## Dockerization through GitHub Actions (CI/CD)
* Created workflow to auto build docker image of the entire repository
* Pushed image to [Docker Repository](https://hub.docker.com/repository/docker/abhinay1289/secom-app) for serving the model

## Cloud Deployment
* Deployed on **AWS(Amazon Web Services) EC2** (Ubuntu instance)
* Pulled Docker image in the instance and served API on cloud

## Try it out
run `docker pull abhinay1289/secom-app:latest`\
run `docker run -p 8000:8000 abhinay1289/secom-app:latest`\

access API on `http://localhost:8000` and interactive docs on `http://localhost:8000/docs`\

*Example response*
```
{
    "prediction": prediction,
    "prob_pass": float(pass_prob),
    "prob_fail": float(fail_prob),
    "threshold":threshold}  # returing threshold for debugging purposes
```
