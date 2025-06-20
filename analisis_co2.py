

from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px

# =====================
# Cargar y preparar datos
# =====================
df = pd.read_csv("co2_pcap_cons.csv")
df_numeric = df.drop(columns='country').apply(pd.to_numeric, errors='coerce')
df_numeric['country'] = df['country']
df_long = df_numeric.melt(id_vars='country', var_name='year', value_name='co2_per_capita')
df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
df_long = df_long.dropna(subset=['co2_per_capita'])

# =====================
# Preparar modelo ML
# =====================
df_ml = df_numeric.melt(id_vars='country', var_name='year', value_name='co2')
df_ml['year'] = df_ml['year'].astype(int)
df_ml.dropna(inplace=True)

le = LabelEncoder()
df_ml['country_encoded'] = le.fit_transform(df_ml['country'])

X = df_ml[['year', 'country_encoded']]
y = df_ml['co2']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# =====================
# Funciones de visualización
# =====================
def show_menu():
    print("\n--- Menú de Análisis de CO₂ ---")
    print("1. Histograma por año")
    print("2. Boxplot por años específicos")
    print("3. Matriz de correlación")
    print("4. Evolución por país")
    print("5. Top 10 países por año")
    print("6. Promedio histórico top 15")
    print("7. Mapa interactivo por año")
    print("8. Predicción LINEAL: país y año (para extrapolar)")
    print("0. Salir")

def plot_histograma(year):
    data = df_numeric[str(year)].dropna()
    plt.figure(figsize=(10, 5))
    sns.histplot(data, bins=30, kde=True)
    plt.title(f'Histograma de CO₂ per cápita ({year})')
    plt.xlabel('Toneladas per cápita')
    plt.tight_layout()
    plt.show()

def plot_boxplot(years):
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_numeric[years])
    plt.title(f'Boxplot CO₂ per cápita ({", ".join(years)})')
    plt.ylabel('Toneladas per cápita')
    plt.tight_layout()
    plt.show()

def plot_correlacion():
    subset = df_numeric[['2000', '2010', '2020', '2022']]
    corr = subset.corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Matriz de correlación")
    plt.tight_layout()
    plt.show()

def plot_evolucion(paises):
    data = df_long[df_long['country'].isin(paises)]
    plt.figure(figsize=(12, 6))
    for pais in paises:
        subset = data[data['country'] == pais]
        plt.plot(subset['year'], subset['co2_per_capita'], label=pais)
    plt.legend()
    plt.title("Evolución CO₂ per cápita")
    plt.xlabel("Año")
    plt.ylabel("Toneladas per cápita")
    plt.tight_layout()
    plt.show()

def plot_top10(year):
    data = df_long[df_long['year'] == year].nlargest(10, 'co2_per_capita')
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, y='country', x='co2_per_capita')
    plt.title(f"Top 10 CO₂ per cápita en {year}")
    plt.tight_layout()
    plt.show()

def plot_promedio_top15():
    mean_data = df_long.groupby('country')['co2_per_capita'].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=mean_data.values, y=mean_data.index)
    plt.title("Top 15 países por promedio histórico")
    plt.tight_layout()
    plt.show()

def plot_mapa_interactivo(year):
    data = df_long[df_long['year'] == year]
    fig = px.choropleth(
        data, locations="country", locationmode="country names",
        color="co2_per_capita", hover_name="country",
        color_continuous_scale="OrRd",
        title=f"Mapa interactivo de CO₂ per cápita ({year})"
    )
    fig.show()

def predecir_lineal():
    pais_usuario = input("🔎 Ingresa el país para predecir (ej. Chile): ")
    año_usuario = int(input("📅 Ingresa el año a predecir (Desde 2023 en adelante): "))
    # Verificar que el año sea mayor al último año registrado
    max_anio = df_ml['year'].max()
    if año_usuario <= max_anio:
        print(f"⚠️ Este modelo está diseñado para predecir desde {max_anio + 1} en adelante.")
        return

    # Filtrar los datos solo del país
    df_pais = df_ml[df_ml['country'] == pais_usuario]

    if df_pais.empty:
        print(f"⚠️ El país '{pais_usuario}' no se encuentra en la base de datos.")
        return
    # Filtrar los datos desde 1980
    df_pais = df_pais[df_pais['year'] >= 1980]


    # Entrenar modelo lineal solo con ese país
    X = df_pais[['year']]
    y = df_pais['co2']
    modelo = LinearRegression()
    modelo.fit(X, y)

    # Predecir
    prediccion = modelo.predict([[año_usuario]])
    print(f"📈 Predicción LINEAL de CO₂ per cápita para {pais_usuario} en {año_usuario}: {round(prediccion[0], 3)} toneladas")


# =====================
# Ejecutar menú
# =====================
if __name__ == "__main__":
    while True:
        show_menu()
        opcion = input("Selecciona una opción: ")

        if opcion == "0":
            print("Saliendo...")
            break
        elif opcion == "1":
            año = input("Año del histograma: ")
            plot_histograma(año)
        elif opcion == "2":
            años = input("Años separados por coma (ej: 2000,2010,2020): ").split(',')
            plot_boxplot([a.strip() for a in años])
        elif opcion == "3":
            plot_correlacion()
        elif opcion == "4":
            paises = input("Países separados por coma (ej: Chile,China): ").split(',')
            plot_evolucion([p.strip() for p in paises])
        elif opcion == "5":
            año = int(input("Año del Top 10: "))
            plot_top10(año)
        elif opcion == "6":
            plot_promedio_top15()
        elif opcion == "7":
            año = int(input("Año del mapa interactivo: "))
            plot_mapa_interactivo(año)
        elif opcion == "8":
             predecir_lineal()
    
        else:
            print("Opción no válida.")
