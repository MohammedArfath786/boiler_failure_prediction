import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV

# Load the dataset from your local machine
file_path = r'C:\Users\moham\OneDrive\Desktop\Boiler_Unit_data_10k.csv'
boiler_data = pd.read_csv(file_path)

# Selecting features and target
features = boiler_data.drop(columns=['Date', 'Device_ID', 'Failure'])  # Drop non-numeric and target columns
target = boiler_data['Failure']

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)

# Creating a pipeline for preprocessing and model training
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler()),  # Scaling features
    ('classifier', RandomForestClassifier())  # Random Forest Classifier
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10, None]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Best model
best_model = grid_search.best_estimator_

# Testing on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save the model to a file for deployment
model_filename = r'C:\Users\moham\OneDrive\Desktop\boiler_failure_model.pkl'
joblib.dump(best_model, model_filename)

st.write(f"Model trained with an accuracy of {accuracy:.2f}")

# Streamlit app for model inference
st.title("Boiler Failure Prediction App")

# Load the trained model
model = joblib.load(model_filename)

# Input fields for the features
Temperature_Boiler_Surface = st.number_input("Temperature Boiler Surface (°C)", min_value=0.0)
Temperature_Exhaust_Gas = st.number_input("Temperature Exhaust Gas (°C)", min_value=0.0)
Temperature_Steam_Output = st.number_input("Temperature Steam Output (°C)", min_value=0.0)
Pressure_Steam = st.number_input("Pressure Steam (bar)", min_value=0.0)
Pressure_Fuel_Line = st.number_input("Pressure Fuel Line (bar)", min_value=0.0)
Pressure_Water = st.number_input("Pressure Water (bar)", min_value=0.0)
Water_Level = st.number_input("Water Level (%)", min_value=0.0)
Vibration_Boiler_Body = st.number_input("Vibration Boiler Body (mm/s x 10^2)", min_value=0.0)
Oxygen_Flue_Gas = st.number_input("Oxygen Flue Gas (% vol)", min_value=0.0)
Carbon_Monoxide_Flue_Gas = st.number_input("Carbon Monoxide Flue Gas (ppm)", min_value=0.0)
Corrosion_Potential = st.number_input("Corrosion Potential (µV)", min_value=0.0)

# Predict button
if st.button("Predict Boiler Failure"):
    # Prepare input for the model
    input_data = np.array([[Temperature_Boiler_Surface, Temperature_Exhaust_Gas, Temperature_Steam_Output,
                            Pressure_Steam, Pressure_Fuel_Line, Pressure_Water, Water_Level, 
                            Vibration_Boiler_Body, Oxygen_Flue_Gas, Carbon_Monoxide_Flue_Gas, Corrosion_Potential]])

    # Make prediction
    prediction = model.predict(input_data)

    # Display the result
    if prediction == 1:
        st.error("Warning: Boiler failure predicted!")
    else:
        st.success("Boiler is operating normally.")
