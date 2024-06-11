import subprocess
import sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Ensure all required packages are installed
required_packages = [
    "pandas", "matplotlib", "seaborn", "scikit-learn",
    "graphviz", "streamlit", "missingno"
]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# Now import the packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
import graphviz
import streamlit as st
import missingno as mn
import io

# Set the page configuration
with st.sidebar:
    st.markdown("**This is bold text in the sidebar**")

# The main page content
st.write("22AIA-WEBSPIRITS.")


# Sidebar for file upload and model selection
st.sidebar.title("Upload and Configure")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Header for the main page
st.title("Naive Bayesian Classifier")

if uploaded_file is not None:
    # Read the dataset
    df = pd.read_csv(uploaded_file)

    # Show the first five rows of the dataset
    st.header("Dataset Preview")
    st.write(df.head())

    # Display dataset information
    st.header("Dataset Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    st.text(s)

    # Find and display missing values
    st.header("Missing Values")
    st.write(df.isna().sum())

    # Visualize missing values
    st.subheader("Missing Values Visualization")
    fig, ax = plt.subplots()
    mn.bar(df, ax=ax)
    st.pyplot(fig)

    # Ensure 'fraud' column exists
    if 'fraud' not in df.columns:
        st.error("The dataset must contain a 'fraud' column as the target variable.")
    else:
        X = df.drop('fraud', axis=1)
        Y = df['fraud']

        st.header("Feature Analysis")
        st.subheader("Boxplot of Features")
        fig, axes = plt.subplots(5, 5, figsize=(20, 20))
        for i, col in enumerate(X.columns):
            if i < 25:  # Adjust according to the number of features
                sns.boxplot(x=X[col], ax=axes[i//5, i%5])
                axes[i//5, i%5].set_xlabel(col, fontsize=20)
        plt.tight_layout()
        st.pyplot(fig)

        # Split X and Y
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        # Standardize the data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Decision Tree classifier
        clf = DecisionTreeClassifier(criterion="entropy", random_state=0)
        clf.fit(X_train, Y_train)

        # Predict the values of testing
        Y_pred_dt = clf.predict(X_test)

        # Calculate and display the confusion matrix
        st.header("Model Performance")
        st.subheader("Decision Tree")
        st.write("Confusion Matrix:")
        cm_dt = confusion_matrix(Y_test, Y_pred_dt)
        st.write(cm_dt)

        # Calculate and display the accuracy score
        accuracy_dt = accuracy_score(Y_test, Y_pred_dt)
        st.write("Accuracy:", accuracy_dt)

        # Visualize the decision tree
        st.write("Decision Tree Visualization:")
        dot_data = export_graphviz(clf, feature_names=X.columns, filled=True, out_file=None)
        st.graphviz_chart(dot_data)

        # Naive Bayes Algorithm
        clf1 = GaussianNB()

        # Train the model
        clf1.fit(X_train, Y_train)

        # Predict the values of testing
        Y_pred_nb = clf1.predict(X_test)

        # Calculate and display the confusion matrix
        st.subheader("Naive Bayes")
        st.write("Confusion Matrix:")
        cm_nb = confusion_matrix(Y_test, Y_pred_nb)
        st.write(cm_nb)

        # Calculate and display the accuracy score
        accuracy_nb = accuracy_score(Y_test, Y_pred_nb)
        st.write("Accuracy:", accuracy_nb)

        # Comparing the accuracy scores
        accuracies = [accuracy_dt * 100, accuracy_nb * 100]
        labels = ['Decision Tree', 'Naive Bayes']

        # Display the accuracy comparison bar chart
        st.subheader("Accuracy Comparison")
        fig, ax = plt.subplots()
        ax.bar(labels, accuracies, color=['blue', 'green'])
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Comparison of Model Accuracies')
        st.pyplot(fig)
else:
    st.info("Please upload a CSV file to proceed.")
