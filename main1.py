import streamlit as st
import pandas as pd
import numpy as np

st.write("Team: 22AIA-WEBSPIRITS")
st.title("Candidate-Elimination Algorithm")


def load_data(file):
    data = pd.read_csv(file)
    return data

def candidate_elimination(data):
    # Extract features and target
    features = data.iloc[:, :-1]
    target = data.iloc[:, -1]
    num_features = features.shape[1]

    # Initialize the most specific hypothesis (S) and most general hypothesis (G)
    S = ['0'] * num_features
    G = [['?'] * num_features]

    for i, instance in features.iterrows():
        if target[i] == 'Yes':  # For positive instances
            for j in range(num_features):
                if S[j] == '0':
                    S[j] = instance[j]
                elif S[j] != instance[j]:
                    S[j] = '?'
            G = [g for g in G if all(g[k] == '?' or g[k] == instance[k] for k in range(num_features))]
        elif target[i] == 'No':  # For negative instances
            G_new = []
            for g in G:
                for j in range(num_features):
                    if g[j] == '?':
                        for value in np.unique(features.iloc[:, j]):
                            if value != instance[j]:
                                g_new = g[:]
                                g_new[j] = value
                                if all(g_new[k] == '?' or g_new[k] == S[k] or S[k] == '0' for k in range(num_features)):
                                    G_new.append(g_new)
            G = G_new
    return S, G

st.write("Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Training Data:")
    st.write(data)

    S, G = candidate_elimination(data)
    
    st.write("Most Specific Hypothesis (S):")
    st.write(S)
    
    st.write("Most General Hypothesis (G):")
    for g in G:
        st.write(g)
