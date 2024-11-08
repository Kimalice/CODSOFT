import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import re

# Load dataset
file_path = 'data/spam.csv'  
data = pd.read_csv(file_path, encoding='ISO-8859-1')

# Keep necessary columns 
data = data[['v1', 'v2']]
data.columns = ['label', 'message']

# Text cleaning function
def clean_text(text):
    text = re.sub(r'\W', ' ', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    return text

# Apply text cleaning
data['message'] = data['message'].apply(clean_text)

# Split dataset into training and testing sets
X = data['message']
y = data['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize messages using TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Predict results on test set
y_pred = model.predict(X_test_tfidf)

# Print accuracy and classification report
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# Prepare data for interactive visualization
results = pd.DataFrame({'Message': X_test, 'Prediction': y_pred})
results['Category'] = results['Prediction'].apply(lambda x: 'Spam' if x == 'spam' else 'Legitimate')

# Create interactive bar chart
fig = px.histogram(results, x='Category', color='Category', 
                   title='Interactive Visualization of SMS Classification',
                   labels={'Category': 'Message Type'},
                   text_auto=True)

fig.update_layout(yaxis_title='Number of Messages', xaxis_title='Category')

# Save the figure as an HTML file
fig.write_html('spam_detection.html')
print("Interactive plot saved as 'spam_detection.html'. You can open it in your web browser.")
