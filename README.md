 Customer Churn Prediction

An end-to-end machine learning project that predicts whether a customer will churn (leave a subscription service). It includes data cleaning, exploratory data analysis, four trained models, hyperparameter tuning, an interactive Streamlit dashboard and a FastAPI REST service.

LIVE DEMO APP LINK :-
https://customer-churn-prediction-112.streamlit.app/
## 📸 Screenshots


| 

![Screenshot 1](Customer%20Churn%20Prediction%20App/UI%20IMAGES/Screenshot%202026-10-04%20150833.png)

 | 

![Screenshot 2](Customer%20Churn%20Prediction%20App/UI%20IMAGES/Screenshot%202026-10-04%20150844%20-%20Copy.png)

 | 

![Screenshot 3](Customer%20Churn%20Prediction%20App/UI%20IMAGES/Screenshot%202026-10-04%20150854.png)

 |
| 

![Screenshot 4](Customer%20Churn%20Prediction%20App/UI%20IMAGES/Screenshot%202026-10-04%20150912.png)

 | 

![Screenshot 5](Customer%20Churn%20Prediction%20App/UI%20IMAGES/Screenshot%202026-10-04%20150924.png)

 | |

## Features

- Data cleaning, EDA and statistical analysis
- Four models trained and compared: Logistic Regression, Decision Tree, Random Forest and KNN
- Hyperparameter tuning of Random Forest with RandomizedSearchCV
- Streamlit dashboard with prediction, probability, risk level, feature importance, data visualization, model performance and multiple model selection
- FastAPI service with single and batch prediction endpoints and automatic API documentation

## Dataset

- File: customer_churn_dataset-training-master.csv
- Source: https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset
- Size: 102,828 rows and 12 columns (102,827 rows after cleaning)
- Target: Churn (1 = churned, 0 = retained)
- Features: Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction
- The dataset is highly imbalanced: about 1.3% retained customers and 98.7% churned customers.

## Project Structure

```
.
├── DATASET/          # Dataset CSV file
├── NOTEBOOK/         # Jupyter notebook (EDA, training, tuning)
├── MODEL FILE/       # Saved model pipeline (modelll.pkl)
├── STREAMLIT/        # Streamlit dashboard (app.py)
├── FAST API/         # FastAPI service (api.py)
├── GRAPH IMAGES/     # EDA and result graphs
├── UI IMAGES/        # Streamlit and API screenshots
├── REPORT/           # Project report
└── requirements.txt  # Python dependencies
```

## Methodology

1. Loaded the dataset and inspected its structure.
2. Cleaned the data: filled missing values, removed the row with a missing target, checked duplicates and outliers (IQR), and dropped CustomerID.
3. Performed EDA and statistical analysis (class distribution, contract length, tenure, correlation heatmap, skewness).
4. Built a preprocessing pipeline: median imputation and StandardScaler for numeric features, most-frequent imputation and OneHotEncoder for categorical features.
5. Split the data 80/20 with random_state = 42.
6. Trained and evaluated four models.
7. Tuned the Random Forest with RandomizedSearchCV (5 iterations, 3-fold CV, ROC-AUC scoring).
8. Saved the final pipeline with joblib.

## Model Results

Before tuning:

- Logistic Regression: Accuracy 0.8723, F1-score 0.9310, ROC-AUC 0.9628
- Decision Tree: Accuracy 0.9990, F1-score 0.9995, ROC-AUC 0.9676
- Random Forest: Accuracy 0.9965, F1-score 0.9982, ROC-AUC 0.9997
- KNN: Accuracy 0.9897, F1-score 0.9948, ROC-AUC 0.8867

After tuning, Random Forest: Accuracy 0.9979, Precision 0.9983, Recall 0.9996, F1-score 0.9989, ROC-AUC 0.9997.

The tuned Random Forest was selected as the final model.

Note: because the dataset is heavily imbalanced, accuracy alone can be misleading. Check the confusion matrix and the recall of the retained class as well.


## Installation

```bash
git clone https://github.com/HamzaWaseem2005/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
python -m pip install -r requirements.txt
```

Important: `scikit-learn==1.6.1` is required because the model was created with this version.

## Run the Streamlit App

```bash
cd STREAMLIT
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## Run the FastAPI Service

```bash
cd "FAST API"
python -m uvicorn api:app --reload
```

Open http://127.0.0.1:8000/docs for the interactive API documentation.

### API Endpoints

- `POST /predict`: predict churn for one customer
- `POST /predict/batch`: predict churn for up to 1,000 customers
- `GET /models`: list available models
- `GET /feature-importance`: top features of a model
- `GET /metrics`: evaluation metrics of the models
- `GET /health`: health check

### Example Request

```json
{
  "age": 35,
  "gender": "Male",
  "tenure": 24,
  "usage_frequency": 15,
  "support_calls": 3,
  "payment_delay": 5,
  "subscription_type": "Basic",
  "contract_length": "Monthly",
  "total_spend": 500,
  "last_interaction": 10
}
```

### Example Response

```json
{
  "model": "Random Forest (Tuned)",
  "prediction": 1,
  "label": "Churn",
  "churn_probability": 0.87,
  "retain_probability": 0.13,
  "risk_level": "High",
  "risk_factors": ["Monthly contract: monthly subscribers showed the highest churn in the EDA."]
}
```

## Technologies Used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, joblib, Streamlit, FastAPI, Uvicorn, Google Colab

## Future Improvements

- Handle class imbalance with SMOTE or decision threshold tuning
- Evaluate with precision-recall curves and cross-validation
- Try gradient boosting models (XGBoost, LightGBM)
- Add SHAP explanations

## Author

Muhammad Hamza
