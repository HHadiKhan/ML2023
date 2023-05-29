import streamlit as st
from PIL import Image
import pickle
import shap
import numpy as np
import lime
from lime import lime_tabular
import dice_ml
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from xgboost import XGBClassifier

st.set_page_config(layout="wide")

data = pd.read_csv("diabetes_prediction_dataset.csv")

smoking_history_mapping = {'not current' : 'former', 'ever' : 'current', 'No Info' : 'unrevealed'}
data['smoking_history'] = data['smoking_history'].replace(smoking_history_mapping)
data['smoking_history'] = data['smoking_history'].apply(lambda x: x.split()[0])

df = pd.get_dummies(data)
y= df['diabetes']
feats= df.drop('diabetes', axis=1)
copied_feats=feats.copy()

pickle_in = open("LGBoost.pkl", "rb")
LGBoost = pickle.load(pickle_in)

pickle_in = open("GradBoost.pkl", "rb")
GradBoost = pickle.load(pickle_in)

pickle_in = open("XGBoost.pkl", "rb")
XGBoost = pickle.load(pickle_in)

pickle_in = open("ADABoost.pkl", "rb")
ADABoost = pickle.load(pickle_in)

age=0
hypertension=0
heart_disease=0
bmi=0
HbA1c_level=0
blood_glucose_level=0
gender_Female=0
gender_Male=0
gender_Other=0
smoking_history_current=0
smoking_history_former=0
smoking_history_never=0
smoking_history_unrevealed=0


st.title('This form will predict your chances of getting Diabetes')
st.write('This is a web app to predict the chances of a patient getting Diabetes. You need to fill in the form in the sidebar, and then the predictor will perform its prediction')

st.sidebar.header("**User Inputs**")
st.sidebar.write("Enter the details for Diabetes Prediction here:")

model_selection = st.sidebar.selectbox('**Select Prediction Model**', ('Gradient Boost', 'LightGBM Boost', 'XG Boost','ADA Boost'), index=0)

age = st.sidebar.slider(
    label='**Age**',
    min_value=2,
    max_value=80,
    value=2,
    step=1
)

genderrad = st.sidebar.radio('**Gender**', ('Male', 'Female', 'Other'))
if genderrad == 'Male':
    gender_Male = 1
elif genderrad == 'Female':
    gender_Female = 1
else:
    gender_Other = 1

bmi = st.sidebar.slider(
    label= 'BMI',
    min_value=10.0,
    max_value=45.0,
    value=10.0,
    step=0.1
)

smoking_historyrad = st.sidebar.radio('**Do you smoke?**', ('Yes', 'Used to', 'No', 'Do not want to share'))
if smoking_historyrad == 'Yes':
    smoking_history_current = 1
elif smoking_historyrad == 'Used to':
    smoking_history_former = 1
elif smoking_historyrad == 'No':
    smoking_history_never = 1
else:
    smoking_history_unrevealed = 1

heart_diseaserad = st.sidebar.radio('**Do you suffer from Heart Disease?**', ('Yes', 'No'))
if heart_diseaserad == 'Yes':
    heart_disease = 1
else:
    heart_disease = 0

hypertension = st.sidebar.radio('**Do you suffer from Hypertension?**', ('Yes', 'No'))
if hypertension == 'Yes':
    hypertension = 1
else:
    hypertension = 0

HbA1c_level = st.sidebar.slider(
    label='**Average Blood Glucose (HbA1c) level**',
    min_value=0.0,
    max_value=10.0,
    value=0.0,
    step=0.1
)

blood_glucose_level = st.sidebar.slider(
    label='**Blood Glucose Level**',
    min_value=80,
    max_value=300,
    value=80,
    step=1
)

features = {
    'age': age,
    'hypertension': hypertension,
    'heart_disease': heart_disease,
    'bmi': bmi,
    'HbA1c_level': HbA1c_level,
    'blood_glucose_level': blood_glucose_level,
    'gender_Female': gender_Female,
    'gender_Male': gender_Male,
    'gender_Other': gender_Other,
    'smoking_history_current': smoking_history_current,
    'smoking_history_former': smoking_history_former,
    'smoking_history_never': smoking_history_never,
    'smoking_history_unrevealed': smoking_history_unrevealed
}

if model_selection == 'ADA Boost':
    model = AdaBoostClassifier(n_estimators=70, learning_rate=1.0, random_state=42)
    model.fit(feats, y)
    prediction = model.predict([list(features.values())])
elif model_selection == 'Gradient Boost':
    model = GradientBoostingClassifier(learning_rate=0.2, max_depth=2, max_features=8, n_estimators=212)
    model.fit(feats, y)
    prediction = model.predict([list(features.values())])
elif model_selection == 'XG Boost':
    model = xgb.XGBClassifier(n_estimators=500, max_depth=2, learning_rate=0.1)
    model.fit(feats, y)
    prediction = model.predict(np.array([list(features.values())]))
elif model_selection == 'LightGBM Boost':
    model = lgb.LGBMClassifier(subsample=1.0, num_leaves=80, min_child_samples=20, max_depth=5, learning_rate=0.1, colsample_bytree=1.0)
    model.fit(feats, y)
    prediction = model.predict([list(features.values())])


def run_lime_prediction():
    explainer = lime.lime_tabular.LimeTabularExplainer(feats.values, feature_names=feats.columns, class_names=['0', '1'])

    # Select an instance from the test data for explanation
    instance = copied_feats.iloc[len(copied_feats)-1]

    # Explain the prediction for the selected instance
    explanation = explainer.explain_instance(instance.values, model.predict_proba, num_features=len(feats.columns))

    # Plot the Lime prediction graph
    fig = explanation.as_pyplot_figure()
    
    # Display the Lime plot in Streamlit using matplotlib's figure
    st.subheader('LIME Prediction')
    st.markdown("**Chosen model:** " + model_selection)
    st.table(copied_feats[len(copied_feats)-1:len(copied_feats)])
    st.pyplot(fig)

def run_shap_prediction():
    import shap

    # Select an input instance for which you want to explain the predictions
    input_instance = copied_feats.iloc[len(copied_feats)-1]  # Replace with your desired input instance

    # Create a SHAP explainer
    explainer = shap.Explainer(model, feats)

    # Generate SHAP values for the input instance
    shap_values = explainer(input_instance)
    shap_values_matrix = np.array([shap_values.values])

    # Plot the SHAP values
    figshap = shap.summary_plot(shap_values_matrix, feature_names=feats.columns)
    st.subheader('SHAP Prediction')
    st.markdown("**Chosen model:** " + model_selection)
    st.table(copied_feats[len(copied_feats)-1:len(copied_feats)])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.pyplot(figshap)


if st.button("Predict"):
    copied_feats = copied_feats.append(features, ignore_index=True)
    st.table(copied_feats[len(copied_feats)-1:len(copied_feats)])
    if prediction == 1:
        st.write('According to the ',model_selection, 'model, you are more likely to have diabetes')
    else:
        st.write('According to the ',model_selection, 'model, you are less likely to have diabetes')

if  st.button("LIME Prediction"):  
    copied_feats = copied_feats.append(features, ignore_index=True)
    run_lime_prediction()

if  st.button("SHAP Prediction"):
    copied_feats = copied_feats.append(features, ignore_index=True)
    if model_selection == 'ADA Boost':
        st.write("SHAP Prediction is not available for ADA Boost model. Please choose Gradient Boosting, XG Boosting or LightGBM Boosting methods for SHAP Prediction")
    else:    
        run_shap_prediction()
