import streamlit as st
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    st.title('KNN Classifier for Iris Dataset')

    # Load Iris dataset
    dataset = load_iris()

    # User input for random state
    random_state = st.sidebar.slider('Random State:', min_value=0, max_value=100, value=0)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=random_state)

    # User input for number of neighbors
    n_neighbors = st.sidebar.slider('Number of Neighbors:', min_value=1, max_value=10, value=1)

    # Train KNN classifier
    kn = KNeighborsClassifier(n_neighbors=n_neighbors)
    kn.fit(X_train, y_train)

    # Prediction and evaluation
    st.write('Prediction Results:')
    for i in range(len(X_test)):
        x = X_test[i]
        x_new = np.array([x])
        prediction = kn.predict(x_new)
        st.write(f"TARGET={y_test[i]} {dataset.target_names[y_test[i]]}, PREDICTED={prediction} {dataset.target_names[prediction]}")

    accuracy = kn.score(X_test, y_test)
    st.write(f'Accuracy: {accuracy:.2f}')

if __name__ == '__main__':
    main()
