# Wine Quality Prediction

Predict wine quality from chemical properties using machine learning.

This repository contains code, experiments, and notebooks for building models that predict the quality of wine (red/white) based on physicochemical measurements. The project focuses on data exploration, feature engineering, model training, evaluation, and basic inference.

## Contents / Purpose
- Provide reproducible code and documentation to train and evaluate machine learning models for wine quality prediction.
- Demonstrate common ML workflow: EDA → preprocessing → modelling → evaluation → inference.
- Serve as a starting point for experimentation with different models (Random Forest, XGBoost, Logistic Regression, etc.), hyperparameter tuning, and feature engineering.

## Dataset
This project typically uses the UCI Wine Quality datasets:
- Wine Quality — Red: winequality-red.csv
- Wine Quality — White: winequality-white.csv

Each record contains physicochemical tests (fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH, sulphates, alcohol) and a quality score between 0 and 10.

Download links:
- UCI: https://archive.ics.uci.edu/ml/datasets/wine+quality

Note: The repository may include a `data/` folder with sample CSVs. If not, download and place the files under `data/`.

## Problem Framing
- Regression: Predict the numeric quality score.
- Classification: Map quality into classes (e.g., low/medium/high or binarize quality >= 7).

Decide the framing in your experiments by adjusting preprocessing and loss/metrics.

## Quickstart (Suggested)
1. Clone the repo
   git clone https://github.com/Maheswaran101/Wine-Quality-Prediction.git
   cd Wine-Quality-Prediction

2. Create a virtual environment and install dependencies
   python3 -m venv .venv
   source .venv/bin/activate    # Windows: .venv\Scripts\activate
   pip install -r requirements.txt

   If the repository does not include requirements.txt, common packages:
   pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib notebook

3. Prepare data
   - Place `winequality-red.csv` and/or `winequality-white.csv` in a `data/` folder.
   - Optionally combine or sample datasets for experiments.

4. Run EDA / Notebook
   jupyter notebook notebooks/exploration.ipynb

5. Train a model (example)
   python src/train.py --data data/winequality-red.csv --target quality --model random_forest --out models/rf_red.pkl

   Note: Replace the command above with the training script that exists in this repo. If the repo uses a different CLI, check the notebook or scripts for exact usage.

6. Predict / Inference (example)
   python src/predict.py --model models/rf_red.pkl --input data/sample.csv --output predictions.csv

## Suggested Repository Structure
- data/                  # raw and processed datasets (not checked in)
- notebooks/             # exploratory notebooks (EDA, modelling experiments)
- src/                   # training, evaluation and inference scripts
- models/                # saved model artifacts
- reports/               # evaluation reports, plots
- requirements.txt       # pip dependencies
- README.md              # project README (this file)

Adjust paths and filenames to match what's included in your copy of the repository.

## Modeling & Evaluation
- Typical models to try:
  - Random Forest / Gradient Boosting (XGBoost, LightGBM)
  - Logistic Regression / SVM (for classification)
  - Linear Regression, Ridge/Lasso (for regression)
- Feature engineering:
  - Scaling (StandardScaler/MinMax)
  - Binning target for classification
  - Interaction terms, polynomial features (if helpful)
- Evaluation metrics:
  - Regression: RMSE, MAE, R^2
  - Classification: Accuracy, Precision, Recall, F1, ROC-AUC (if binary)
- Cross-validation and hyperparameter tuning recommended (GridSearchCV / RandomizedSearchCV).

## Example Results (illustrative)
Results will vary by preprocessing, model and hyperparameters. Example metrics from a typical random forest baseline:
- Regression (RMSE): ~0.6–0.9
- Classification (accuracy, 3-class): ~55–70%

These are illustrative; run your experiments to get actual values.

## Reproduce Experiments
- Use fixed random seeds where appropriate.
- Log experiments (e.g., with MLflow, weights & biases, or simple CSV logs).
- Save preprocessing pipelines (scikit-learn Pipeline) together with model artifacts so inference is reproducible.

## Contributing
Contributions are welcome. Suggested ways to contribute:
- Add missing scripts for training/inference with clear CLI.
- Provide reproducible notebooks & example runs.
- Add
