import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Cargar el modelo entrenado
modelo_rf = joblib.load('modelo_entrenado.pkl')

# Título de la aplicación
st.title("Aplicación de Predicción de Regresión")

# Descripción de la aplicación
st.write("""
Esta aplicación predice valores usando un modelo de regresión entrenado.
""")

# Entrada de datos del usuario
st.header("Ingrese los datos para la predicción:")

# Definir los nombres de las características reales
feature_names = [
    'gearboxbearingtemperature', 'gearboxoiltemperature', 'generatorrpm',
    'generatorwinding1temperature', 'generatorwinding2temperature',
    'reactivepower', 'rotorrpm', 'winddirection', 'windspeed'
]

# Crear un formulario para las características
input_data = {}
for feature in feature_names:
    input_data[feature] = st.number_input(f'Ingrese {feature}', value=0.0)

# Organizar las características en un DataFrame
df_input = pd.DataFrame([input_data])

# Botón para hacer la predicción
if st.button('Predecir'):
    prediccion = modelo_rf.predict(df_input)
    st.subheader("Predicción:")
    st.write(prediccion[0])

# Ejecutar la aplicación con:
# streamlit run app.py



