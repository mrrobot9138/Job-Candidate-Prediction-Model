# Job Role Prediction Project

## Overview
This project uses a Decision Tree Regressor to predict the suitability of individuals for different job roles based on various attributes such as age, self-description, skills, and education specialization. The application is built using Python and Streamlit for an interactive interface.

## Data Generation
A dataset of 200 individuals is generated with random attributes including:
- Name
- Age and Age Category
- Self Description
- Job Title
- Skills
- Education Specialization

## Data Preprocessing
1. Encode categorical features using OneHotEncoder and LabelEncoder.
2. Combine encoded features with numerical data.

## Model Training
A Decision Tree Regressor is trained on the processed dataset to predict job suitability.

## Streamlit Interface
The application allows users to select a job role from a sidebar dropdown. It then filters the dataset based on the selected job role and predicts the suitability of individuals.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/job-role-prediction.git
    ```
2. Navigate to the project directory:
    ```bash
    cd job-role-prediction
    ```
3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
Run the Streamlit application:
```bash
streamlit run app.py
```

## Files
- `generate_data.py`: Script to generate the dataset.
- `app.py`: Main application file.
- `requirements.txt`: List of dependencies.

## Future Work
- Improve model accuracy with more sophisticated algorithms.
- Add more features for better predictions.
- Enhance the user interface.

## License
This project is licensed under the MIT License.
