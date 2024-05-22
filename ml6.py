import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def main():
    st.title('Sentiment Analysis with Naive Bayes Classifier')
    
    # Provide the full path to the CSV file
    file_path = r"C:\Users\MOORTHY\Downloads\document.csv"
    
    # Attempt to load the data
    try:
        msg = pd.read_csv(file_path, names=['message', 'label'])
    except FileNotFoundError:
        st.error(f"Error: File '{file_path}' not found.")
        return
    
    st.write("Total Instances of Dataset:", msg.shape[0])
    msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

    # Split data into train and test sets
    X = msg.message
    y = msg.labelnum
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y)  # Pass y as well
    
    # Vectorize the text data
    count_v = CountVectorizer()
    Xtrain_dm = count_v.fit_transform(Xtrain)
    Xtest_dm = count_v.transform(Xtest)

    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(Xtrain_dm, ytrain)
    pred = clf.predict(Xtest_dm)

    # Display predictions and evaluation metrics
    st.write('Sample Predictions:')
    for doc, p in zip(Xtest, pred):
        p = 'pos' if p == 1 else 'neg'
        st.write(f"{doc} -> {p}")

    st.write('\nAccuracy Metrics:')
    st.write('Accuracy:', accuracy_score(ytest, pred))
    st.write('Recall:', recall_score(ytest, pred))
    st.write('Precision:', precision_score(ytest, pred))
    st.write('Confusion Matrix:\n', confusion_matrix(ytest, pred))

if __name__ == '__main__':
    main()
