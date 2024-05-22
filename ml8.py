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

# Plotting
def plot_clusters(X, y, predY, y_cluster_gmm):
    colormap = np.array(['red', 'lime', 'black'])

    # Real Plot
    plt.subplot(1, 3, 1)
    plt.scatter(X[:, 2], X[:, 3], c=colormap[y], s=40)
    plt.title('Real')

    # KMeans Plot
    plt.subplot(1, 3, 2)
    plt.scatter(X[:, 2], X[:, 3], c=colormap[predY], s=40)
    plt.title('KMeans')

    # GMM Plot
    plt.subplot(1, 3, 3)
    plt.scatter(X[:, 2], X[:, 3], c=colormap[y_cluster_gmm], s=40)
    plt.title('GMM Classification')

    st.pyplot()

# Main function
def main():
    st.title('Clustering Visualization')

    # Plotting
    plt.figure(figsize=(14, 7))

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
