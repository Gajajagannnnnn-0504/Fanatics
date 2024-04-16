import streamlit as st
import numpy as np

class CandidateElimination:
    def __init__(self, num_attributes):
        # Initialize the most specific and most general hypotheses
        self.S = ['0'] * num_attributes  # Most specific hypothesis
        self.G = ['?'] * num_attributes  # Most general hypothesis

    def eliminate(self, X, y):
        for i in range(len(X)):
            x = X[i]
            if y[i] == 'Yes':  # Positive instance
                self.eliminate_negative(x)
            else:  # Negative instance
                self.eliminate_positive(x)

    def eliminate_positive(self, x):
        # Remove inconsistent hypotheses from S
        for i in range(len(self.S)):
            if self.S[i] != x[i]:
                self.S[i] = '?' if self.S[i] != '?' else x[i]
        
        # Refine G
        for i in range(len(self.G)):
            if self.G[i] != '?' and self.G[i] != x[i]:
                self.G[i] = '?'

    def eliminate_negative(self, x):
        # Remove inconsistent hypotheses from G
        for i in range(len(self.G)):
            if self.G[i] != '?' and self.G[i] != x[i]:
                self.G[i] = '?' if self.S[i] == '?' else self.S[i]

    def print_hypotheses(self):
        st.write("Most specific hypothesis (S):", ''.join(self.S))
        st.write("Most general hypothesis (G):", ''.join(self.G))


def main():
    st.title("Candidate Elimination Algorithm")

    num_instances = st.number_input("Enter the number of instances:", min_value=1, step=1)
    num_attributes = st.number_input("Enter the number of attributes:", min_value=1, step=1)
import streamlit as st
import numpy as np

class CandidateElimination:
    def __init__(self, num_attributes):
        # Initialize the most specific and most general hypotheses
        self.S = ['0'] * num_attributes  # Most specific hypothesis
        self.G = ['?'] * num_attributes  # Most general hypothesis

    def eliminate(self, X, y):
        for i in range(len(X)):
            x = X[i]
            if y[i] == 'Yes':  # Positive instance
                self.eliminate_negative(x)
            else:  # Negative instance
                self.eliminate_positive(x)

    def eliminate_positive(self, x):
        # Remove inconsistent hypotheses from S
        for i in range(len(self.S)):
            if self.S[i] != x[i]:
                self.S[i] = '?' if self.S[i] != '?' else x[i]
        
        # Refine G
        for i in range(len(self.G)):
            if self.G[i] != '?' and self.G[i] != x[i]:
                self.G[i] = '?'

    def eliminate_negative(self, x):
        # Remove inconsistent hypotheses from G
        for i in range(len(self.G)):
            if self.G[i] != '?' and self.G[i] != x[i]:
                self.G[i] = '?' if self.S[i] == '?' else self.S[i]

    def print_hypotheses(self):
        st.write("Most specific hypothesis (S):", ''.join(self.S))
        st.write("Most general hypothesis (G):", ''.join(self.G))


def main():
    st.title("Candidate Elimination Algorithm")

    num_instances = st.number_input("Enter the number of instances:", min_value=1, step=1)
    num_attributes = st.number_input("Enter the number of attributes:", min_value=1, step=1)

    X = []
    y = []

    for i in range(num_instances):
        instance = []
        st.write(f"Instance {i + 1}:")
        for j in range(num_attributes):
            attribute = st.selectbox(f"Select attribute {j + 1} value for instance {i + 1}:", ['Yes', 'No'])
            instance.append(attribute)
        label = st.radio(f"Select label for instance {i + 1}:", ['Yes', 'No'])
        X.append(instance)
        y.append(label)

    if st.button("Run Algorithm"):
        ce = CandidateElimination(num_attributes=num_attributes)
        ce.eliminate(X, y)
        ce.print_hypotheses()


    X = []
    y = []

    for i in range(num_instances):
        instance = []
        st.write(f"Instance {i + 1}:")
        for j in range(num_attributes):
            attribute = st.selectbox(f"Select attribute {j + 1} value for instance {i + 1}:", ['Yes', 'No'])
            instance.append(attribute)
        label = st.radio(f"Select label for instance {i + 1}:", ['Yes', 'No'])
        X.append(instance)
        y.append(label)

    if st.button("Run Algorithm"):
        ce = CandidateElimination(num_attributes=num_attributes)
        ce.eliminate(X, y)
        ce.print_hypotheses()

import streamlit as st
import numpy as np

class CandidateElimination:
    def __init__(self, num_attributes):
        # Initialize the most specific and most general hypotheses
        self.S = ['0'] * num_attributes  # Most specific hypothesis
        self.G = ['?'] * num_attributes  # Most general hypothesis

    def eliminate(self, X, y):
        for i in range(len(X)):
            x = X[i]
            if y[i] == 'Yes':  # Positive instance
                self.eliminate_negative(x)
            else:  # Negative instance
                self.eliminate_positive(x)

    def eliminate_positive(self, x):
        # Remove inconsistent hypotheses from S
        for i in range(len(self.S)):
            if self.S[i] != x[i]:
                self.S[i] = '?' if self.S[i] != '?' else x[i]
        
        # Refine G
        for i in range(len(self.G)):
            if self.G[i] != '?' and self.G[i] != x[i]:
                self.G[i] = '?'

    def eliminate_negative(self, x):
        # Remove inconsistent hypotheses from G
        for i in range(len(self.G)):
            if self.G[i] != '?' and self.G[i] != x[i]:
                self.G[i] = '?' if self.S[i] == '?' else self.S[i]

    def print_hypotheses(self):
        st.write("Most specific hypothesis (S):", ''.join(self.S))
        st.write("Most general hypothesis (G):", ''.join(self.G))


def main():
    st.title("Candidate Elimination Algorithm")

    num_instances = st.number_input("Enter the number of instances:", min_value=1, step=1)
    num_attributes = st.number_input("Enter the number of attributes:", min_value=1, step=1)

    X = []
    y = []

    for i in range(num_instances):
        instance = []
        st.write(f"Instance {i + 1}:")
        for j in range(num_attributes):
            attribute = st.selectbox(f"Select attribute {j + 1} value for instance {i + 1}:", ['Yes', 'No'])
            instance.append(attribute)
        label = st.radio(f"Select label for instance {i + 1}:", ['Yes', 'No'])
        X.append(instance)
        y.append(label)

    if st.button("Run Algorithm"):
        ce = CandidateElimination(num_attributes=num_attributes)
        ce.eliminate(X, y)
        ce.print_hypotheses()

if __name__ == "__main__":
    main()
