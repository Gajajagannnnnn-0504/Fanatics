import streamlit as st
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

def main():
    st.title('Heart Disease Prediction')

    # Load the dataset
    try:
        data = pd.read_csv(r"C:\Users\MOORTHY\Downloads\heartdisease.csv")
        heart_disease = pd.DataFrame(data)
    except FileNotFoundError:
        st.error("Error: Dataset 'heartdisease.csv' not found. Make sure the file exists and is in the correct directory.")
        return
    except Exception as e:
        st.error(f"An error occurred while loading the dataset: {e}")
        return

    # Define the Bayesian Network model
    model = BayesianModel([
        ('age', 'Lifestyle'),
        ('Gender', 'Lifestyle'),
        ('Family', 'heartdisease'),
        ('diet', 'cholestrol'),
        ('Lifestyle', 'diet'),
        ('cholestrol', 'heartdisease'),
        ('diet', 'cholestrol')
    ])

    # Fit the model using Maximum Likelihood Estimator
    model.fit(heart_disease, estimator=MaximumLikelihoodEstimator)

    # Create a VariableElimination object for inference
    HeartDisease_infer = VariableElimination(model)

    # Display user input fields
    st.write('Enter the following values for prediction:')
    age = st.number_input('Age', min_value=0, max_value=100, step=1)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    family_history = st.selectbox('Family History', ['Yes', 'No'])
    diet = st.selectbox('Diet', ['High', 'Medium'])
    lifestyle = st.selectbox('Lifestyle', ['Athlete', 'Active', 'Moderate', 'Sedentary'])
    cholesterol = st.selectbox('Cholesterol', ['High', 'Borderline', 'Normal'])

    # Convert user inputs to appropriate values for inference
    gender = 0 if gender == 'Male' else 1
    family_history = 1 if family_history == 'Yes' else 0
    diet = 0 if diet == 'High' else 1
    lifestyle_map = {'Athlete': 0, 'Active': 1, 'Moderate': 2, 'Sedentary': 3}
    lifestyle = lifestyle_map[lifestyle]
    cholesterol_map = {'High': 0, 'Borderline': 1, 'Normal': 2}
    cholesterol = cholesterol_map[cholesterol]

    # Perform inference
    try:
        q = HeartDisease_infer.query(variables=['heartdisease'], evidence={
            'age': age,
            'Gender': gender,
            'Family': family_history,
            'diet': diet,
            'Lifestyle': lifestyle,
            'cholestrol': cholesterol
        })

        # Display prediction result
        if q:
            max_prob_state = q.values.argmax()  # Get the index of the state with the highest probability
            st.write('Prediction:', max_prob_state)
        else:
            st.write('No prediction available.')

    except Exception as e:
        st.error(f"An error occurred during inference: {e}")

if __name__ == '__main__':
    main()
