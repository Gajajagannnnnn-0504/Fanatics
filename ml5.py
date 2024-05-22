import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

# Define a function to load and preprocess data
def load_data(uploaded_file):
    try:
        # Read the contents of the uploaded file
        file_buffer = io.BytesIO(uploaded_file.getvalue())
        data = pd.read_csv(file_buffer)
        if data.empty:
            st.error("Error: Uploaded file is empty.")
            return None, None
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        # Convert categorical variables to numerical
        label_encoders = {}
        for column in X.columns:
            le = LabelEncoder()
            X[column] = le.fit_transform(X[column])
            label_encoders[column] = le
        y = LabelEncoder().fit_transform(y)
        return X, y
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        return None, None

# Main function to run the Streamlit app
def main():
    st.title('Tennis Game Prediction')
    
    # Sidebar to upload file
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.write("### First 5 rows of data:")
        # Pass the uploaded file object to the load_data function
        X, y = load_data(uploaded_file)
        if X is not None and y is not None:
            st.dataframe(X.head())
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
            
            classifier = GaussianNB()
            classifier.fit(X_train, y_train)
            
            y_pred = classifier.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.write("### Model Accuracy:")
            st.write(f"Accuracy of the model: {accuracy:.2f}")

if __name__ == '__main__':
    main()
