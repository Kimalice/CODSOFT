# Importing necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load the dataset
# Make sure to adjust the path if needed
data = pd.read_csv('data/churn_data.csv')

# Displaying the first few rows and info of the dataset
print(data.head())
print(data.info())

# Exploring the target variable distribution
sns.countplot(x='Exited', data=data)
plt.title('Churn Distribution (Exited = 1 means Churned)')
plt.show()

# Preprocessing: Dropping unnecessary columns
data.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

# Converting categorical variables into dummy/indicator variables
data = pd.get_dummies(data, drop_first=True)

# Splitting the dataset into features and target variable
X = data.drop('Exited', axis=1)  # Features
y = data['Exited']  # Target variable (Churn)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better performance of algorithms
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

# Predictions
y_pred_log = log_reg.predict(X_test)

# Evaluating the Logistic Regression model
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_log)}")
print("Classification Report for Logistic Regression:")
print(classification_report(y_test, y_pred_log))

# Confusion Matrix for Logistic Regression
confusion_mtx_log = confusion_matrix(y_test, y_pred_log)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx_log, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Logistic Regression')
plt.show()

# Random Forest Classifier Model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)

# Predictions with Random Forest
y_pred_rf = rf_classifier.predict(X_test)

# Evaluating the Random Forest model
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))

# Confusion Matrix for Random Forest
confusion_mtx_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_mtx_rf, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Churned', 'Churned'], yticklabels=['Not Churned', 'Churned'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix for Random Forest')
plt.show()
