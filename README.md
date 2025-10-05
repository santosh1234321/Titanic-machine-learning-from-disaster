# Titanic Survival Prediction
## Project Description
This project tackles the classic Kaggle "Titanic: Machine Learning from Disaster" challenge. The goal is to predict the survival of passengers aboard the Titanic using machine learning techniques. The solution implements data preprocessing, feature engineering, and hyperparameter-tuned Random Forest classification to achieve robust predictive performance. This repository demonstrates a clean, reproducible pipeline from raw data to predictions, suitable for learning and further model improvement.
## Project Structure
- main.py : Core Python script running the full pipeline
- titanic.csv : Training data (not included, download from Kaggle)
- test.csv : Test data (not included, download from Kaggle)
- predictions.csv : Generated prediction results after running the script
## Features
- Mean imputation of missing Ages
- One-hot encoding of categorical variables (Sex, Embarked)
- Feature dropping of irrelevant columns like Name, Ticket, and Cabin
- Stratified train-validation split for balanced training
- Random Forest model with hyperparameter tuning via grid search
- Standardized numeric features for improved model performance
## Results
The model achieves approximately 77% accuracy on the validation split and provides a reliable baseline for Kaggle submission. Further improvements can be made by enhancing feature engineering and exploring ensemble models.
