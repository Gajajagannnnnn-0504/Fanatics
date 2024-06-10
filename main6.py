import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

st.title("22AIA-WEBSPIRITS")
# Title and introduction
st.title("Bayesian Network for COVID-19 Symptom Classification")
st.write("This app uses a Bayesian Network to classify COVID-19 symptoms based on the provided dataset.")

# Create a synthetic dataset
def create_synthetic_data():
    np.random.seed(42)
    size = 1000
    data = {
        'Fever': np.random.randint(2, size=size),
        'cough': np.random.randint(2, size=size),
        'runnynose': np.random.randint(2, size=size),
        'sorethroat': np.random.randint(2, size=size),
        'pain': np.random.randint(2, size=size),
        'diarrhoea': np.random.randint(2, size=size),
        'diffbreath': np.random.randint(2, size=size),
        'nose': np.random.randint(2, size=size),
        'target': np.random.randint(2, size=size),
        'tired': np.random.randint(2, size=size),
    }
    df = pd.DataFrame(data)
    return df

# Generate the synthetic dataset
df = create_synthetic_data()

# Display the first five rows of the dataset
st.subheader("First Five Rows of the Dataset")
st.write(df.head())

# Dataset description
st.subheader("Dataset Description")
st.write(df.describe())

# Check for missing values
st.subheader("Missing Values")
st.write(df.isna().sum())

# Display histograms for all numeric columns
st.subheader("Histograms")
for column in df.select_dtypes(include=np.number).columns:
    st.write(f"Histogram for {column}")
    fig, ax = plt.subplots()
    ax.hist(df[column].dropna(), bins=20)
    st.pyplot(fig)

# Define the Bayesian Network structure
st.subheader("Bayesian Network Structure")
model = BayesianModel([
    ('Fever', 'tired'),
    ('cough', 'sorethroat'),
    ('pain', 'runnynose'),
    ('diarrhoea', 'target'),
    ('diffbreath', 'Fever'),
    ('nose', 'target'),
    ('target', 'cough'),
    ('sorethroat', 'pain'),
    ('Fever', 'target')
])
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Display CPD values
st.subheader("Conditional Probability Distributions (CPDs)")
for node in model.nodes():
    st.write(f"CPD of {node}")
    st.text(model.get_cpds(node))

# Perform Variable Elimination for inference
st.subheader("Inference using Variable Elimination")
infer = VariableElimination(model)

symptoms = ['Fever', 'cough', 'runnynose']
evidence = {}
for symptom in symptoms:
    value = st.selectbox(f"Do you have {symptom}?", ('Yes', 'No', 'Not Sure'), key=symptom)
    if value == 'Yes':
        evidence[symptom] = 1
    elif value == 'No':
        evidence[symptom] = 0

if evidence:
    target_result = infer.query(variables=["target"], evidence=evidence)
    st.write("Probability of having COVID-19 given the symptoms:")
    st.write(target_result)
else:
    st.write("Please provide evidence for symptoms to get a prediction.")
