import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

# Function to load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to apply K-means clustering
def kmeans_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(data)
    return kmeans.labels_

# Function to apply EM clustering
def em_clustering(data, num_clusters):
    em = GaussianMixture(n_components=num_clusters)
    em.fit(data)
    return em.predict(data)

# Function to visualize clusters
def visualize_clusters(data, labels, algorithm):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
    plt.title(f'Clustered Data ({algorithm})')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar(label='Cluster')
    st.pyplot()

def main():
    st.title('Clustering Comparison: K-means vs EM')

    st.sidebar.header('Settings')
    file_path = st.sidebar.file_uploader('Upload CSV File', type=['csv'])
    if file_path is not None:
        data = load_data(file_path)
        st.write('Sample data:')
        st.write(data.head())

        num_clusters = st.sidebar.slider('Number of Clusters', min_value=2, max_value=10, value=3, step=1)
        run_clustering = st.sidebar.button('Run Clustering')

        if run_clustering:
            data_array = data.values
            kmeans_labels = kmeans_clustering(data_array, num_clusters)
            em_labels = em_clustering(data_array, num_clusters)

            st.write('K-means Labels:')
            st.write(kmeans_labels)
            st.write('EM Labels:')
            st.write(em_labels)

            visualize_clusters(data_array, kmeans_labels, 'K-means')
            visualize_clusters(data_array, em_labels, 'EM')

if __name__ == '__main__':
    main()
