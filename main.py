import csv
import random
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# Sample data for generating random values
names = ['Alice', 'Bob', 'Charlie', 'David', 'Emma', 'Frank', 'Grace', 'Henry', 'Ivy', 'Jack']
self_descriptions = ['Hardworking', 'Creative', 'Detail-oriented', 'Team player', 'Analytical', 'Adaptable']
job_titles = ['Software Engineer', 'Data Scientist', 'Marketing Manager', 'Sales Representative', 'Teacher']
skills = ['Python', 'Data Analysis', 'Marketing Strategy', 'Communication', 'Problem Solving']
education_specializations = ['Computer Science', 'Business Administration', 'Psychology', 'Engineering']
previous_salaries = ['Low', 'Medium', 'High']
age_categories = ['Young', 'Mature', 'Senior']

# Generate data for 200 people
data = []
for _ in range(200):
    name = random.choice(names)
    age = random.randint(20, 60)
    age_category = 'Young' if age <= 35 else ('Mature' if age <= 50 else 'Senior')
    self_description = random.choice(self_descriptions)
    job_title = random.choice(job_titles)
    num_skills = random.randint(1, len(skills))
    skill = random.sample(skills, num_skills)  # Random number of skills
    education_specialization = random.sample(education_specializations, random.randint(1, len(education_specializations)))  # Random number of education specializations
    row = [name, age, age_category, self_description, job_title, skill, education_specialization]
    data.append(row)

# Write data to a CSV file
with open('people_data.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Name', 'Age', 'Age Category', 'Self Description', 'Job Title', 'Skills', 'Education Specialization'])
    writer.writerows(data)

# Read data from the CSV file
data = pd.read_csv('people_data.csv')

# Preprocessing the data
label_encoder = LabelEncoder()
data['Job Title Encoded'] = label_encoder.fit_transform(data['Job Title'])
X_categorical = data[['Self Description', 'Skills', 'Education Specialization', 'Age Category']]
X_encoded = pd.get_dummies(X_categorical)  # One-hot encoding categorical features
X = pd.concat([data['Age'], X_encoded], axis=1)
y = data['Job Title Encoded']

# Train the decision tree regressor
regressor = DecisionTreeRegressor()
regressor.fit(X, y)

# Streamlit interface
st.title("Job Role Prediction")
st.sidebar.title("Select Job Role")

# Dropdown to select job role
selected_job = st.sidebar.selectbox("Select Job Role", job_titles)

# Display selected job role
st.sidebar.markdown(f"Selected Job Role: *{selected_job}*")

# Filter data based on selected job role
filtered_data = data[data['Job Title'] == selected_job]
X_categorical_filtered = filtered_data[['Self Description', 'Skills', 'Education Specialization', 'Age Category']]
X_encoded_filtered = pd.get_dummies(X_categorical_filtered)  # One-hot encoding categorical features

# Ensure that the filtered data has all the columns present in the training data
missing_cols = set(X.columns) - set(X_encoded_filtered.columns)
for col in missing_cols:
    X_encoded_filtered[col] = 0

# Reorder the columns to match the order in the training data
X_filtered = X_encoded_filtered[X.columns]

# Make predictions for the filtered data
y_pred = regressor.predict(X_filtered)

# Scale predictions to the range [1, 100]
min_val = np.min(y_pred)
max_val = np.max(y_pred)
scaled_predictions = ((y_pred - min_val) / (max_val - min_val)) * 100

# Display prediction results
st.subheader("Prediction Results")
st.write("Prediction Results:")
prediction_results = pd.DataFrame({
    'Name': filtered_data['Name'],
    'Predicted Suitability': scaled_predictions.round(1)
})
st.write(prediction_results)
