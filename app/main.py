
import streamlit as st
import pandas as pd
import joblib

# Load the trained Decision Tree model and the scaler
# Make sure these files are in the same directory as your Streamlit app or provide the full path
try:
    model = joblib.load('decision_tree_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Model or scaler files not found. Please ensure 'decision_tree_model.pkl' and 'scaler.pkl' are in the same directory.")
    st.stop()

# Set the title of the Streamlit app
st.title('Iris Species Prediction App')
st.write('Enter the measurements of the iris flower to predict its species.')

# Create input fields for the four features
sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.5)
sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0)
petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0)
petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.2)

# Create a button to make predictions
if st.button('Predict Species'):
    # Create a DataFrame from the user inputs
    input_data = pd.DataFrame([{
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }])

    # Scale the input data using the loaded scaler
    scaled_input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(scaled_input_data)
    
    st.success(f'The predicted Iris species is: **{prediction[0]}**')
