# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Title of the Streamlit app
st.title("Diabetes Prediction using SVM")

# Upload dataset (assume the file is uploaded as CSV)
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])

if uploaded_file is not None:
    # Load the dataset
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    st.write("Dataset:")
    st.write(df.head())

    # Show outcome distribution
    st.write("Outcome distribution:")
    st.write(df['Outcome'].value_counts())

    # Splitting the data
    input_data = df.drop(columns='Outcome', axis=1)
    labels = df['Outcome']

    # Data Standardization
    scaler = StandardScaler()
    scaler.fit(input_data)
    standardized_data = scaler.transform(input_data)

    # Train Test Split
    x_train, x_test, y_train, y_test = train_test_split(standardized_data, labels, 
                                                        test_size=0.2, stratify=labels, random_state=2)
    
    # Model training (SVM Classifier)
    classifier_linear = svm.SVC(C=1.0, kernel='linear')
    classifier_linear.fit(x_train, y_train)

    # Training accuracy
    x_pred = classifier_linear.predict(x_train)
    train_accuracy = accuracy_score(x_pred, y_train)
    st.write(f'Accuracy score of the training data: {train_accuracy:.2f}')

    # Testing accuracy
    y_pred = classifier_linear.predict(x_test)
    test_accuracy = accuracy_score(y_test, y_pred)
    st.write(f'Accuracy score of the testing data: {test_accuracy:.2f}')
    
    # Prediction on new data (user input)
    st.write("Enter new data for prediction:")
    
    # Creating input fields for new data
    feature_names = df.columns[:-1]  # Assuming all except 'Outcome' are features
    input_values = []
    for feature in feature_names:
        input_val = st.number_input(f"Enter value for {feature}", step=1.0)
        input_values.append(input_val)
    
    if st.button("Predict"):
        new_data = np.asarray([input_values])
        std_data = scaler.transform(new_data)

        # Prediction
        new_pred = classifier_linear.predict(std_data)
        if new_pred[0] == 0:
            st.success('The person is non-diabetic')
        else:
            st.error('The person is diabetic')
