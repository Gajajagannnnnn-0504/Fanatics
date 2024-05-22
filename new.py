import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def preprocess_data(data):
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target variable
    return X, y

def train_naive_bayes(X_train, y_train):
    nb_classifier = GaussianNB()
    nb_classifier.fit(X_train, y_train)
    return nb_classifier

def test_accuracy(classifier, X_test, y_test):
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

def main():
    st.title("Naive Bayes Classifier")

    # Upload Training Data
    st.header("Training Data")
    uploaded_train_file = st.file_uploader("Upload CSV file for training data:", type=["csv"])
    if uploaded_train_file is not None:
        train_data = load_data(uploaded_train_file)
        st.write(train_data.head())

        # Preprocess Training Data
        X_train, y_train = preprocess_data(train_data)

        # Train Naive Bayes Classifier
        nb_classifier = train_naive_bayes(X_train, y_train)
        st.success("Naive Bayes Classifier trained successfully!")

    # Upload Test Data
    st.header("Test Data")
    uploaded_test_files = st.file_uploader("Upload CSV files for test data:", type=["csv"], accept_multiple_files=True)
    if uploaded_test_files:
        for uploaded_test_file in uploaded_test_files:
            test_data = load_data(uploaded_test_file)
            st.write(test_data.head())

            # Preprocess Test Data
            X_test, y_test = preprocess_data(test_data)

            # Test Accuracy
            accuracy = test_accuracy(nb_classifier, X_test, y_test)
            st.write("Accuracy:", accuracy)

if __name__ == "__main__":
    main()