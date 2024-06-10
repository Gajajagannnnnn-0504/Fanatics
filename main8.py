import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

df = load_data()

# Split the data into features and target
X = df.drop(columns=['target'])
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the k-Nearest Neighbors classifier
n_neighbors = st.sidebar.slider('Number of neighbors', min_value=1, max_value=10, value=5)
knn = KNeighborsClassifier(n_neighbors=n_neighbors)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)

# Print classification report
st.write('## Classification Report')
st.write(classification_report(y_test, y_pred, target_names=load_iris().target_names))

# Streamlit app
st.title('k-Nearest Neighbors Classifier for Iris Dataset')

st.write('## Dataset')
st.write(df.head())

st.write('## Model Performance')
st.write(f'Accuracy: {accuracy:.4f}')

st.write('## Sample Predictions')
correct_predictions = []
wrong_predictions = []

for i in range(len(y_pred)):
    if y_pred[i] == y_test.iloc[i]:
        correct_predictions.append((X_test.iloc[i], y_test.iloc[i], y_pred[i]))
    else:
        wrong_predictions.append((X_test.iloc[i], y_test.iloc[i], y_pred[i]))

st.write('### Correct Predictions')
for features, true_label, predicted_label in correct_predictions:
    st.write(f'Features: {features.values}, True Label: {load_iris().target_names[true_label]}, Predicted Label: {load_iris().target_names[predicted_label]}')

st.write('### Wrong Predictions')
for features, true_label, predicted_label in wrong_predictions:
    st.write(f'Features: {features.values}, True Label: {load_iris().target_names[true_label]}, Predicted Label: {load_iris().target_names[predicted_label]}')
