import streamlit as st
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv(r"./diabetes.csv")

st.markdown("""
    <style>
        body {
            background-color: #2e3b4e; /* Dark slate gray background */
            font-family: 'Arial', sans-serif;
            color: #f5f5f5;
            margin: 0;
            padding: 0;
        }
        .header {
            font-size: 48px;
            font-family: 'Georgia', serif;
            font-weight: bold;
            text-align: center;
            color: #ff6f61;
            margin-top: 50px;
            padding-bottom: 10px;
            border-bottom: 4px solid #ff6f61;
        }
        .sub-header {
            font-size: 24px;
            font-family: 'Georgia', serif;
            color: #f1c40f;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 40px;
        }
        .highlight {
            font-size: 20px;
            color: #1abc9c;
            font-weight: bold;
            text-align: center;
        }
        hr {
            border: none;
            border-top: 3px solid #1abc9c;
            margin-top: 20px;
            margin-bottom: 20px;
        }
        .table-container {
            background-color: #34495e;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
        }
        .stButton>button {
            background-color: #f39c12;
            color: white;
            font-size: 20px;
            padding: 15px;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            width: 100%;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #e67e22;
            transform: scale(1.05);
        }
        .sidebar .sidebar-content {
            background-color: #e74c3c;
            padding: 20px;
            border-radius: 12px;
        }
        .stSidebar {
            background-color: #e67e22;
            padding: 15px;
            border-radius: 12px;
        }
        .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            color: #fff;
        }
        .sidebar input {
            margin-bottom: 12px;
        }
        .input-label {
            color: #fff;
            font-weight: bold;
        }
    </style>
    <div class="header">Diabetes Risk Checker</div>
    <div class="sub-header">Enter your health details to analyze your diabetes risk.</div>
    <hr>
""", unsafe_allow_html=True)

st.sidebar.title("Patient Information")
st.sidebar.markdown("""
    <style>
        .sidebar .sidebar-header {
            font-size: 20px;
            font-weight: bold;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

st.sidebar.header("Enter Your Health Details:")

def get_user_input():
    pregnancies = st.sidebar.number_input('Pregnancies', min_value=0, max_value=17, value=3, step=1)
    bp = st.sidebar.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=122, value=70, step=1)
    bmi = st.sidebar.number_input('BMI (Body Mass Index)', min_value=0.0, max_value=67.0, value=20.0, step=0.1)
    glucose = st.sidebar.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=120, step=1)
    skinthickness = st.sidebar.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
    dpf = st.sidebar.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.4, value=0.47, step=0.01)
    insulin = st.sidebar.number_input('Insulin Level (IU/mL)', min_value=0, max_value=846, value=79, step=1)
    age = st.sidebar.slider('Age (years)', min_value=21, max_value=88, value=33)

    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': bp,
        'SkinThickness': skinthickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
    }

    features = pd.DataFrame(user_data, index=[0])
    return features

user_data = get_user_input()

st.markdown("<h2 style='color: #f5f5f5;'>Health Data Overview</h2>", unsafe_allow_html=True)

st.markdown("<div class='table-container'>", unsafe_allow_html=True)
st.table(user_data)
st.markdown("</div>", unsafe_allow_html=True)

x = df.drop(['Outcome'], axis=1)
y = df['Outcome']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

rf = RandomForestClassifier()
rf.fit(x_train, y_train)

if st.button('Analyze Risk'):
    st.markdown("<h3 style='text-align: center; color: #f1c40f;'>Analyzing your health data...</h3>", unsafe_allow_html=True)
    
    progress = st.progress(0)
    for percent in range(100):
        progress.progress(percent + 1)
    
    prediction = rf.predict(user_data)
    
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #f5f5f5;'>Prediction Result</h2>", unsafe_allow_html=True)
    result = 'You are not diabetic.' if prediction[0] == 0 else 'You are at risk of diabetes.'
    st.markdown(f"<p class='highlight'>{result}</p>", unsafe_allow_html=True)
    
    accuracy = accuracy_score(y_test, rf.predict(x_test)) * 100
    st.markdown(f"<p style='color: #f1c40f; font-size: 18px;'>Model Accuracy: {accuracy:.2f}%</p>", unsafe_allow_html=True)

else:
    st.markdown("<h3 style='text-align: center; color: #f1c40f;'>Enter your data and click 'Analyze Risk'</h3>", unsafe_allow_html=True)
