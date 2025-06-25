import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib

# Cargar datos
df = pd.read_csv("co2.csv")  # Asegúrate de tener este archivo

# Preprocesamiento
df_long = df.melt(id_vars="country", var_name="year", value_name="co2")
df_long.dropna(inplace=True)
df_long["year"] = df_long["year"].astype(int)

# Codificar países como números
df_long["country_code"] = df_long["country"].astype("category").cat.codes

# Variables predictoras y target
X = df_long[["country_code", "year"]]
y = df_long["co2"]

# División de datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar modelo
joblib.dump(model, "modelo_rf.pkl")

# Predicciones y métricas
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Guardar métricas
with open("resultados.txt", "w") as f:
    f.write(f"RMSE: {rmse:.4f}\n")
    f.write(f"Media: {np.mean(y):.2f}\n")
    f.write(f"Desviación estándar: {np.std(y):.2f}\n")
    f.write(f"Mediana: {np.median(y):.2f}\n")

# Función para limpiar archivos
def limpiar_archivos():
    for archivo in ["resultados.txt", "modelo_rf.pkl"]:
        if os.path.exists(archivo):
            os.remove(archivo)
    print("Archivos eliminados.")

# Predicción futura
def predecir_pais_anio():
    pais = input("Ingresa el país: ")
    anio = int(input("Ingresa el año: "))

    # Cargar modelo
    model = joblib.load("modelo_rf.pkl")
    
    # Obtener el código del país
    if pais in df_long["country"].values:
        cod = df_long[df_long["country"] == pais]["country_code"].values[0]
        X_new = pd.DataFrame([[cod, anio]], columns=["country_code", "year"])
        pred = model.predict(X_new)[0]
        print(f"Predicción de CO₂ para {pais} en {anio}: {pred:.2f} toneladas per cápita")
    else:
        print("País no encontrado.")

# Menú principal
def menu():
    while True:
        print("\nMenú:")
        print("1. Entrenar modelo")
        print("2. Ver métricas")
        print("3. Predecir CO₂ por país y año")
        print("4. Limpiar archivos")
        print("5. Salir")

        opcion = input("Selecciona una opción: ")

        if opcion == "1":
            print("Modelo ya entrenado.")
        elif opcion == "2":
            with open("resultados.txt", "r") as f:
                print(f.read())
        elif opcion == "3":
            predecir_pais_anio()
        elif opcion == "4":
            limpiar_archivos()
        elif opcion == "5":
            break
        else:
            print("Opción inválida.")

# Ejecutar menú
menu()
