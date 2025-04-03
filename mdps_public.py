import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# Load the saved models with error handling
try:
    diabetes_model = pickle.load(open('diabetes_model.sav', 'rb'))
    heart_disease_model = pickle.load(open('heart_disease_model.sav', 'rb'))
    parkinsons_model = pickle.load(open('parkinsons_model.sav', 'rb'))
except Exception as e:
    st.error(f"Error loading models: {e}")

# Sidebar navigation
with st.sidebar:
    selected = option_menu(
        'Multiple Disease Prediction System',
        ['Diabetes Prediction', 'Heart Disease Prediction', "Parkinson's Prediction"],
        icons=['activity', 'heart', 'person'],
        default_index=0
    )

# Diabetes Prediction Page
if selected == 'Diabetes Prediction':
    st.title('Diabetes Prediction using ML')
    with st.form("diabetes_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            Pregnancies = int(st.number_input('Number of Pregnancies', min_value=0))
        with col2:
            Glucose = float(st.number_input('Glucose Level'))
        with col3:
            BloodPressure = float(st.number_input('Blood Pressure value'))
        with col1:
            SkinThickness = float(st.number_input('Skin Thickness value'))
        with col2:
            Insulin = float(st.number_input('Insulin Level'))
        with col3:
            BMI = float(st.number_input('BMI value'))
        with col1:
            DiabetesPedigreeFunction = float(st.number_input('Diabetes Pedigree Function value'))
        with col2:
            Age = int(st.number_input('Age of the Person', min_value=0))
        submit = st.form_submit_button("Diabetes Test Result")
        if submit:
            result = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
            st.success('The person is diabetic' if result[0] == 1 else 'The person is not diabetic')

# Heart Disease Prediction Page
if selected == 'Heart Disease Prediction':
    st.title('Heart Disease Prediction using ML')
    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            age = int(st.number_input('Age', min_value=0))
        with col2:
            sex = int(st.number_input('Sex (0: Female, 1: Male)', min_value=0, max_value=1))
        with col3:
            cp = int(st.number_input('Chest Pain types', min_value=0, max_value=3))
        with col1:
            trestbps = float(st.number_input('Resting Blood Pressure'))
        with col2:
            chol = float(st.number_input('Serum Cholesterol in mg/dl'))
        with col3:
            fbs = int(st.number_input('Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)', min_value=0, max_value=1))
        with col1:
            restecg = int(st.number_input('Resting Electrocardiographic results', min_value=0, max_value=2))
        with col2:
            thalach = float(st.number_input('Maximum Heart Rate achieved'))
        with col3:
            exang = int(st.number_input('Exercise Induced Angina (1: Yes, 0: No)', min_value=0, max_value=1))
        with col1:
            oldpeak = float(st.number_input('ST depression induced by exercise'))
        with col2:
            slope = int(st.number_input('Slope of the peak exercise ST segment', min_value=0, max_value=2))
        with col3:
            ca = int(st.number_input('Major vessels colored by fluoroscopy', min_value=0, max_value=4))
        with col1:
            thal = int(st.number_input('Thal (0: normal, 1: fixed defect, 2: reversible defect)', min_value=0, max_value=2))
        submit = st.form_submit_button("Heart Disease Test Result")
        if submit:
            result = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
            st.success('The person has heart disease' if result[0] == 1 else 'The person does not have heart disease')

# Parkinson's Prediction Page
if selected == "Parkinson's Prediction":
    st.title("Parkinson's Disease Prediction using ML")
    with st.form("parkinsons_form"):
        inputs = []
        labels = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
                  'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3',
                  'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2',
                  'D2', 'PPE']
        for i in range(0, len(labels), 3):
            cols = st.columns(3)
            for j in range(3):
                if i + j < len(labels):
                    with cols[j]:
                        inputs.append(float(st.number_input(labels[i + j])))
        submit = st.form_submit_button("Parkinson's Test Result")
        if submit:
            result = parkinsons_model.predict([inputs])
            st.success("The person has Parkinson's disease" if result[0] == 1 else "The person does not have Parkinson's disease")
