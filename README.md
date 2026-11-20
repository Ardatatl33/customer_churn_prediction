# Customer Churn Prediction

This beginner-friendly machine learning project predicts whether a telecom customer will leave the service (`Churn = Yes`) or stay (`Churn = No`). It explores the data, prepares it for modeling, and compares three classification models.

## Dataset

The project uses the [Telco Customer Churn dataset on Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn). The downloaded CSV is included at `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` so the project can run from a fresh clone. The dataset contains 7,043 rows and 21 columns.

The target is `Churn`. Available customer fields are `customerID`, `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `tenure`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, and `TotalCharges`. The `customerID` field is excluded from model inputs.

## Run the project locally

1. Install Python 3 and open a terminal in the project folder.
2. Install the libraries:

   ```bash
   python -m pip install pandas numpy matplotlib seaborn scikit-learn notebook
   ```

3. Start Jupyter Notebook:

   ```bash
   jupyter notebook
   ```

4. Open `notebooks/customer_churn_prediction.ipynb` and run the cells from top to bottom.

The notebook reads the CSV from the local `data` folder. When opened in Google Colab, it reads the same CSV from this public GitHub repository.

## Models and results

The notebook compares Logistic Regression, Decision Tree, and Random Forest using accuracy, precision, recall, F1 score, and confusion matrices. Churn (`Yes`, encoded as `1`) is treated as the positive class.

| Model | Accuracy | Precision | Recall | F1 score |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.806 | 0.657 | 0.559 | 0.604 |
| Decision Tree | 0.722 | 0.478 | 0.497 | 0.488 |
| Random Forest | 0.788 | 0.628 | 0.492 | 0.552 |

Logistic Regression had the highest recall and F1 score in this run. Its confusion matrix was `[[926, 109], [165, 209]]`; the matrix rows are actual classes (`0`, `1`) and columns are predicted classes (`0`, `1`). These are the results produced by the notebook with the recorded 80/20 stratified split and `random_state=42`.

## Run in Google Colab

[Open the project notebook in Google Colab](https://colab.research.google.com/github/Ardatatl33/customer_churn_prediction/blob/main/notebooks/customer_churn_prediction.ipynb).

## Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Jupyter Notebook.
