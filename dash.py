import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import pickle

image = Image.open("./midia/heart.jpg")

st.set_page_config(
    page_title = "Heart Attack"

)

st.markdown("# Heart Attack Probability")
st.image(image, caption = "Heart")

age = st.number_input("Idade: ", min_value = 0, max_value = 80)

sex = st.selectbox("Sexo: ", ['male', 'female'])

chestpain = st.selectbox("Tipo de dor no peito: ", 
        ['non_ang_pain', 'ang_atypic', 'ang_typic', 'asyntomatic'])

st.sidebar.markdown(">ang_typic: Angina típica consiste em uma dor retroesternal em aperto ou queimação, que dura apenas alguns minutos, piora com esforço ou estresse emocional e é aliviada com repouso ou nitroglicerina.")
st.sidebar.markdown(">ang_atypic: Angina atípica (p. ex., com meteorismo, flatulência, desconforto abdominal) pode ocorrer em alguns pacientes. Esses pacientes frequentemente descrevem os sintomas para indigestão; eructação pode dar sensação de alívio dos sintomas.")
st.sidebar.markdown(">non_ang_pain: Dor não anginosa uma dor no peito é muito provavelmente não anginosa se sua duração for superior a 30 minutos ou inferior a 5 segundos, aumenta com a inspiração, pode ser provocada com um movimento do tronco ou braço, pode ser provocada por pressão local dos dedos ou inclinação para a frente , ou pode ser aliviado imediatamente ao deitar.")
st.sidebar.markdown(">asyntomatic: Assintomático ocorre quando o coração não consegue receber sangue e oxigênio suficientes, mas sem que a pessoa sinta sintoma algum como ocorre nos dois tipos anteriores. Esse tipo de angina ocorre quando se pratica exercícios sem se alongar e aquecer os músculos primeiramente e quando se tem diabetes.")

bloodpressure = st.number_input("Pressão Sanguínea: ", min_value = 90, max_value = 200)

cholesterol = st.number_input("Colesterol: ", min_value = 120, max_value = 600)

glycemia = st.selectbox("Glicemia em jejum > 120 mg/dl(1 - verdadeiro, 0 - falso): ", [1, 0])

maxheartrate = st.number_input("Frequência cardíaca maxima alcançada: ", min_value = 70, max_value = 210)

angex = st.selectbox("Angina causada por exercício: (0 - não, 1 - sim)", [0, 1])

d = {
    'Age': [age],
    'Sex': [sex],
    'ChestPainType': [chestpain],
    'BloodPressureRest': [bloodpressure],
    'Cholesterol': [cholesterol],
    'Glycemia': [glycemia],
    'MaxHeartRate': [maxheartrate],
    'AnginaEx': [angex]
}

df = pd.DataFrame(d)
st.dataframe(df)

with open(r"./models/model.pickle", "rb") as input_file:
    model = pickle.load(input_file)

r = model.predict_proba(df)
proba = r[:, 1][0]

if proba < 0.5:
    st.success(f"Heart attack proba: {np.round(proba, 2)}")
else:
    st.error(f"Heart attack proba: {np.round(proba, 2)}")
