# Loan Default Prediction Model

This project implements an artificial neural networks model to predict the likelihood of loan default at the time of application. The model uses various applicant attributes and loan characteristics to make predictions, helping assess credit risk during the application process.

## Project Overview

The model predicts whether a loan applicant will default (bad_flag = 1) or not (bad_flag = 0) based on various features including:
- Annual income
- Debt-to-income ratio
- Credit utilization metrics
- Employment history
- Home ownership status
- Recent credit inquiries
- And other relevant financial indicators

## Repository Structure

```
LOAN_DEFAULT/
├── Notebooks/
│   ├── eda.ipynb                     # Exploratory Data Analysis
│   ├── model_training.ipynb          # Model Training Pipeline
│   └── test_data_predictions.ipynb   # Generate Predictions on Test Data
├── src/
│   ├── data/
│   │   ├── cleaned_data.csv         # Processed Dataset
│   │   ├── testing_loan_data.csv    # Test Dataset
│   │   └── training_loan_data.csv   # Training Dataset
│   └── Model Results/
│       ├── best_model.pth           # Saved Model Weights
│       ├── preprocessor.pkl         # Saved Preprocessor Pipeline
│       └── test_predictions.csv     # Model Predictions
├── .gitignore
└── README.md
```
- The dataset has been removed due to data safety issues. You can save the data you have with the above naming convention.

## Dependencies

To run this project, you'll need the following dependencies:

### Main Libraries
- Python 3.8+
- pandas
- numpy
- torch (PyTorch)

### Machine Learning & Data Processing
- scikit-learn
- imbalanced-learn
- torch.nn
- torch.optim
- torch.utils.data

### Data Visualization
- seaborn
- matplotlib

### Development Tools
- jupyter

You can install all required packages using:
```bash
pip install pandas numpy torch scikit-learn imbalanced-learn seaborn matplotlib jupyter
```

Or create a new environment using conda:
```bash
conda create -n loan_default python=3.8
conda activate loan_default
conda install pandas numpy pytorch scikit-learn seaborn matplotlib jupyter
pip install imbalanced-learn
```


## How to Run

1. **Data Exploration**:
   ```bash
   jupyter notebook Notebooks/eda.ipynb
   ```
   This notebook contains exploratory data analysis and insights about the features.

2. **Model Training**:
   ```bash
   jupyter notebook Notebooks/model_training.ipynb
   ```
   This notebook implements the model training pipeline, including:
   - Data preprocessing
   - Feature engineering
   - Model training
   - Model evaluation
   - Saving the best model

3. **Generate Predictions**:
   ```bash
   jupyter notebook Notebooks/test_data_predictions.ipynb
   ```
   This notebook loads the trained model and generates predictions for the test dataset.

## Model Details & Assumptions

### Assumptions
1. All monetary values are in the same currency and have been adjusted for inflation
2. Missing values in critical fields have been handled appropriately
3. The training data is representative of the population the model will be used on
4. The relationship between features and default probability is relatively stable over time

### Model Output
The model generates binary predictions:
- 0: Loan is predicted to perform well
- 1: Loan is predicted to default

## Performance Metrics

Model performance metrics and validation results can be found in the `model_training.ipynb` notebook, including:
- ROC-AUC Score
- Precision-Recall curves
- Confusion Matrix
- Feature Importance Analysis
