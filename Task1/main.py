import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Sample data creation
data = {
    'plot': [
        "A young boy discovers he's a wizard.",
        "A romantic love story set in Paris.",
        "A group of friends goes on a dangerous adventure.",
        "A superhero fights against evil forces.",
        "An animated journey through a magical world."
    ],
    'genre': [0, 1, 2, 3, 4]  # Numeric labels for different genres
}

df = pd.DataFrame(data)

# Splitting the dataset into train and test sets
train_data, test_data, train_labels, test_labels = train_test_split(
    df['plot'], df['genre'], test_size=0.2, random_state=42
)

# Text vectorization
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(train_data)
X_test = vectorizer.transform(test_data)

# Model training
model = LogisticRegression()
model.fit(X_train, train_labels)

# Making predictions
predictions = model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(test_labels, predictions)
report = classification_report(test_labels, predictions, output_dict=True)

# Displaying results
print("Accuracy:", accuracy)
print("\nClassification Report:")
print(report)

# Creating a confusion matrix
cm = confusion_matrix(test_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Genre 0', 'Genre 1', 'Genre 2', 'Genre 3', 'Genre 4'],
            yticklabels=['Genre 0', 'Genre 1', 'Genre 2', 'Genre 3', 'Genre 4'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()

#  Printing the report in a more readable format
def print_classification_report(report):
    print("\n----- Classification Report -----")
    for label, metrics in report.items():
        if label not in ['accuracy', 'macro avg', 'weighted avg']:
            print(f"Genre {label}:")
            print(f"  Precision: {metrics['precision']:.2f}")
            print(f"  Recall: {metrics['recall']:.2f}")
            print(f"  F1-score: {metrics['f1-score']:.2f}")
            print(f"  Support: {metrics['support']}")
    print("Accuracy: {:.2f}".format(report['accuracy']))

# Print the detailed report
print_classification_report(report)
