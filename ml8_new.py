import streamlit as st
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt

# Load iris dataset
dataset = load_iris()
X = dataset.data
y = dataset.target

# Plotting function
def plot_clusters(X, y, predY, y_cluster_gmm):
    colormap = np.array(['red', 'lime', 'black'])

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))

    # Real Plot
    axes[0].scatter(X[:, 2], X[:, 3], c=colormap[y], s=40)
    axes[0].set_title('Real')

    # KMeans Plot
    axes[1].scatter(X[:, 2], X[:, 3], c=colormap[predY], s=40)
    axes[1].set_title('KMeans')

    # GMM Plot
    axes[2].scatter(X[:, 2], X[:, 3], c=colormap[y_cluster_gmm], s=40)
    axes[2].set_title('GMM Classification')

    st.pyplot(fig)

# Main function
def main():
    st.title('Clustering Visualization')

    # KMeans clustering
    model = KMeans(n_clusters=3)
    model.fit(X)
    predY = model.labels_

    # Gaussian Mixture Model (GMM) clustering
    scaler = preprocessing.StandardScaler()
    xsa = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=3)
    gmm.fit(xsa)
    y_cluster_gmm = gmm.predict(xsa)

    plot_clusters(X, y, predY, y_cluster_gmm)

if __name__ == '__main__':
    main()
