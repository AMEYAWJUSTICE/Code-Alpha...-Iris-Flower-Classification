# Iris Species Prediction Project

## Project Overview
This project focuses on building a machine learning model to classify Iris flower species based on their sepal and petal measurements. The process involves data loading, cleaning, exploratory data analysis (EDA), data preprocessing, model training, evaluation, and deployment using a Streamlit web application.

## Dataset
The project uses the classic Iris dataset, which contains 150 samples of Iris flowers, each with four features (sepal length, sepal width, petal length, and petal width) and one target variable (species: Iris-setosa, Iris-versicolor, or Iris-virginica).

## Key Steps
1.  **Data Loading and Initial Inspection**: The Iris dataset was loaded and inspected for initial understanding.
2.  **Data Cleaning**: Duplicate rows were identified and removed. No missing values were found.
3.  **Exploratory Data Analysis (EDA)**: Visualizations such as pairplots, feature distribution histograms, and correlation heatmaps were generated to understand data patterns and relationships. Summary statistics were also generated per species.
4.  **Data Preprocessing**: Features (measurements) and labels (species) were separated. The dataset was split into training and testing sets (80/20 ratio). Feature scaling was applied using `StandardScaler`.
5.  **Model Training**: Two classification models, Decision Tree Classifier and Random Forest Classifier, were trained on the preprocessed data.
6.  **Model Evaluation**: Both models were evaluated using accuracy scores, classification reports, and confusion matrices. Both achieved an accuracy of approximately 93.3% on the test set.
7.  **Model Export**: The trained Decision Tree model (`decision_tree_model.pkl`) and the `StandardScaler` object (`scaler.pkl`) were exported for later use in the Streamlit application.
8.  **Streamlit Web Application**: A Streamlit application (`main.py`) was created to provide an interactive interface for predicting Iris species using the exported model and scaler.

## How to Run the Streamlit App
To run the Streamlit web application, follow these steps:

1.  **Ensure Files are Present**: Make sure you have the following files in the same directory:
    *   `decision_tree_model.pkl` (the exported Decision Tree model)
    *   `scaler.pkl` (the exported StandardScaler)
    *   `main.py` (the Streamlit app script generated)
    *   `requirements.txt` (listing all Python dependencies)

2.  **Install Dependencies**: Open your terminal or command prompt, navigate to the directory containing these files, and install the required Python packages using pip:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the App**: Run the Streamlit application using the command:
    ```bash
    streamlit run main.py
    ```

4.  **Access the App**: Your web browser will automatically open to the Streamlit application, where you can input the measurements and get real-time predictions.
