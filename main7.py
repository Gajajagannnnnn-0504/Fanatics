import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

# Load the dataset
@st.cache
def load_data():
    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df

df = load_data()

# Extract features for clustering
X = df.drop(columns=['target'])

# Apply Gaussian Mixture Model (EM algorithm)
gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X)
df['GMM Cluster'] = gmm_labels

# Apply k-Means algorithm
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_labels = kmeans.fit_predict(X)
df['kMeans Cluster'] = kmeans_labels

# Calculate silhouette scores
gmm_silhouette = silhouette_score(X, gmm_labels)
kmeans_silhouette = silhouette_score(X, kmeans_labels)

# Streamlit app
st.title('Clustering with EM Algorithm and k-Means')

st.write('## Dataset')
st.write(df.head())

st.write('## Silhouette Scores')
st.write(f'GMM Silhouette Score: {gmm_silhouette:.4f}')
st.write(f'k-Means Silhouette Score: {kmeans_silhouette:.4f}')

st.write('## Clustering Results')

# Plot the clustering results using Plotly
fig1 = px.scatter_matrix(df, dimensions=df.columns[:-3], color='GMM Cluster', 
                         title='GMM Clustering Results', symbol='target')
fig2 = px.scatter_matrix(df, dimensions=df.columns[:-3], color='kMeans Cluster', 
                         title='k-Means Clustering Results', symbol='target')

st.plotly_chart(fig1)
st.plotly_chart(fig2)

st.write('## Note')
st.write('In this example, we used the Iris dataset. For a real-world application, consider using more comprehensive and current data, and tuning the parameters of the clustering algorithms for better results.')
