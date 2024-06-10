import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

st.write("Team: 22AIA-WEBSPIRITS")
st.title("Clustering Comparison: EM (GMM) vs k-Means")


@st.cache
def load_data(file):
    data = pd.read_csv(file)
    return data

def plot_clusters(data, labels, algorithm_name):
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[:, 0], y=data[:, 1], hue=labels, palette='viridis', marker='o', edgecolor='k')
    plt.title(f'Clusters by {algorithm_name}')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    st.pyplot(plt)

def perform_clustering(data, num_clusters):
    # k-Means Clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(data)
    kmeans_silhouette = silhouette_score(data, kmeans_labels)

    # EM Clustering (Gaussian Mixture Model)
    gmm = GaussianMixture(n_components=num_clusters, random_state=42)
    gmm_labels = gmm.fit_predict(data)
    gmm_silhouette = silhouette_score(data, gmm_labels)

    return kmeans_labels, kmeans_silhouette, gmm_labels, gmm_silhouette

st.write("Upload your CSV file")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    st.write("Dataset:")
    st.write(data.head())

    feature_columns = st.multiselect("Select feature columns for clustering", data.columns, default=data.columns[:2])
    num_clusters = st.slider("Number of clusters", min_value=2, max_value=10, value=3)

    if st.button("Perform Clustering"):
        data_selected = data[feature_columns].values
        
        kmeans_labels, kmeans_silhouette, gmm_labels, gmm_silhouette = perform_clustering(data_selected, num_clusters)
        
        st.write(f"k-Means Silhouette Score: {kmeans_silhouette:.2f}")
        plot_clusters(data_selected, kmeans_labels, "k-Means")

        st.write(f"EM (GMM) Silhouette Score: {gmm_silhouette:.2f}")
        plot_clusters(data_selected, gmm_labels, "EM (GMM)")

        st.write("Comparison:")
        if kmeans_silhouette > gmm_silhouette:
            st.write("k-Means clustering resulted in a better silhouette score, indicating better-defined clusters.")
        else:
            st.write("EM (GMM) clustering resulted in a better silhouette score, indicating better-defined clusters.")