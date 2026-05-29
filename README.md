# Medical Insurance Premium Predictor

A machine learning web app that predicts annual medical insurance premiums
based on personal health and demographic data, built using Random Forest
Regression and deployed with Streamlit.

## Live Demo

[Click here to try the app](https://gaurav-ml2-project.streamlit.app/)

## Problem Statement

Insurance companies use complex models to price premiums. This project builds
an interpretable ML model that estimates an individual's annual insurance cost
based on age, BMI, smoking status, region, gender, and number of dependents.

## Project Structure

insurance-premium-predictor/
├── data/
│   └── insurance.csv
├── model/
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── images/
├── notebooks/
├── app.py
├── requirements.txt
└── README.md

## ML Approach

| Model             | RMSE  | MAE   | R2 Score |
|-------------------|-------|-------|----------|
| Linear Regression | ~6100 | ~4200 | ~0.74    |
| Ridge Regression  | ~6050 | ~4150 | ~0.75    |
| Random Forest     | ~4500 | ~2700 | ~0.87    |

Random Forest was selected as the best model with 87% accuracy.

## Key Features

- Exploratory Data Analysis with distribution plots and heatmaps
- Feature Engineering: BMI category and Age group
- Comparison of 3 regression models on RMSE, MAE and R2
- Interactive web app with live prediction and risk assessment

## Dataset

Source: Kaggle Medical Cost Personal Dataset
Rows: 1338
Features: age, sex, bmi, children, smoker, region
Target: charges (annual insurance cost in USD)

## Run Locally

git clone https://github.com/YOUR_USERNAME/insurance-premium-predictor.git
cd insurance-premium-predictor
pip install -r requirements.txt
streamlit run app.py

## Tech Stack

- Python 3.10+
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Joblib
- Streamlit

## Author

Gaurav Kumar
Email: gk8131831@gmail.com
GitHub: https://github.com/gaurav01011978

## License

This project is licensed under the MIT License.