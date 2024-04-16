import streamlit as st
import numpy as np
from collections import Counter

class Node:
    def __init__(self, attribute=None, label=None):
        self.attribute = attribute  # Attribute used for splitting
        self.label = label  # Class label if it's a leaf node
        self.children = {}  # Dictionary to store child nodes

def entropy(data):
    # Calculate entropy of a dataset
    labels = [row[-1] for row in data]
    label_counts = Counter(labels)
    entropy = 0
    for count in label_counts.values():
        probability = count / len(labels)
        entropy -= probability * np.log2(probability)
    return entropy

def information_gain(data, attribute_index):
    # Calculate information gain of splitting the dataset on a particular attribute
    total_entropy = entropy(data)
    attribute_values = set([row[attribute_index] for row in data])
    weighted_entropy = 0
    for value in attribute_values:
        subset = [row for row in data if row[attribute_index] == value]
        subset_entropy = entropy(subset)
        weighted_entropy += (len(subset) / len(data)) * subset_entropy
    return total_entropy - weighted_entropy

def majority_label(data):
    # Return the majority class label in a dataset
    labels = [row[-1] for row in data]
    return Counter(labels).most_common(1)[0][0]

def id3(data, attributes):
    # Base cases
    if len(set([row[-1] for row in data])) == 1:
        # If all instances have the same class label, return a leaf node
        return Node(label=data[0][-1])
    if len(attributes) == 0:
        # If there are no more attributes to split on, return a leaf node with majority label
        return Node(label=majority_label(data))
    
    # Find the best attribute to split on
    gains = [information_gain(data, i) for i in range(len(attributes))]
    best_attribute_index = np.argmax(gains)
    best_attribute = attributes[best_attribute_index]
    
    # Create a new node with the best attribute
    node = Node(attribute=best_attribute)
    
    # Recursively construct subtree for each value of the best attribute
    attribute_values = set([row[best_attribute_index] for row in data])
    for value in attribute_values:
        subset = [row[:-1] for row in data if row[best_attribute_index] == value]
        child_attributes = attributes[:best_attribute_index] + attributes[best_attribute_index+1:]
        child_node = id3(subset, child_attributes)
        node.children[value] = child_node
    
    return node

def predict(tree, instance):
    # Predict the class label for a single instance using the decision tree
    if tree.label is not None:
        # If the node is a leaf node, return the class label
        return tree.label
    attribute_value = instance[tree.attribute]
    if attribute_value not in tree.children:
        # If the attribute value is not seen during training, return majority label
        return majority_label([instance])
    child_node = tree.children[attribute_value]
    return predict(child_node, instance)

def main():
    st.title("Decision Tree Streamlit App")

    st.sidebar.header("Training Data")

    data = []
    num_attributes = st.sidebar.number_input("Number of Attributes:", min_value=1, value=4)
    for i in range(num_attributes):
        attribute_name = st.sidebar.text_input(f"Attribute {i+1} Name:")
        if attribute_name:
            st.sidebar.write(f"Attribute {i+1} Values:")
            attribute_values = st.sidebar.text_area(f"Attribute {i+1} Values (comma-separated):").split(',')
            for value in attribute_values:
                class_label = st.sidebar.selectbox(f"Class Label for {value}:", options=["Yes", "No"])
                data.append([attribute_name, value, class_label])

    if st.sidebar.button("Train"):
        attributes = list(range(num_attributes))  # Indices of attributes
        tree = id3(data, attributes)
        st.success("Training completed!")

        st.subheader("Decision Tree:")
        display_tree(tree)

        st.subheader("Predictions:")
        new_instance = []
        for i in range(num_attributes):
            attribute_value = st.text_input(f"Attribute {i+1} Value:")
            new_instance.append(attribute_value)
        prediction = predict(tree, new_instance)
        st.write("Prediction:", prediction)

def display_tree(node, depth=0):
    if node.label is not None:
        st.write(" " * depth * 4, f"Class Label: {node.label}")
    else:
        st.write(" " * depth * 4, f"Attribute: {node.attribute}")
        for value, child_node in node.children.items():
            st.write(" " * (depth * 4 + 4), f"Value: {value}")
            display_tree(child_node, depth + 1)

if __name__ == "__main__":
    main()
