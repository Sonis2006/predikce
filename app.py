import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Klimatická predikce Brno", layout="wide")
st.title("🌡️🌧️💨 Historický vývoj a predikce klimatu v Brně")

# === 1. Načtení dat ===
@st.cache_data
def load_climate_data():
    years = list(range(1900, 2024))
    temperature = np.linspace(8.0, 10.5, len(years)) + np.random.normal(0, 0.2, len(years))
    precipitation = np.random.normal(600, 100, len(years))  # mm ročně
    wind_speed = np.random.normal(3.5, 0.5, len(years))     # m/s
    return pd.DataFrame({
        "year": years,
        "temperature": temperature,
        "precipitation": precipitation,
        "wind_speed": wind_speed
    })

df = load_climate_data()

# === 2. Vizualizace historických dat ===
def plot_historical(column, title, color, ylabel):
    st.subheader(title)
    fig, ax = plt.subplots()
    ax.plot(df["year"], df[column], color=color)
    ax.set_xlabel("Rok")
    ax.set_ylabel(ylabel)
    st.pyplot(fig)

plot_historical("temperature", "🌡️ Roční průměr teplot", "blue", "°C")
plot_historical("precipitation", "🌧️ Roční srážky", "green", "mm")
plot_historical("wind_speed", "💨 Průměrná rychlost větru", "orange", "m/s")

# === 3. Predikce pomocí numpy.polyfit ===
def predict_future(df, column, horizons):
    x = df["year"].values
    y = df[column].values
    coef = np.polyfit(x, y, 1)
    future_years = np.array([x.max() + h for h in horizons])
    predictions = coef[0] * future_years + coef[1]
    return future_years, predictions, coef

horizons = [10, 100, 1000]

for column, name, unit, color in [
    ("temperature", "Teplota", "°C", "blue"),
    ("precipitation", "Srážky", "mm", "green"),
    ("wind_speed", "Vítr", "m/s", "orange")
]:
    years_pred, values_pred, coef = predict_future(df, column, horizons)
    
    # Výpis predikcí
    st.subheader(f"📈 Predikce {name}")
    for y, v in zip(years_pred, values_pred):
        st.write(f"Rok {int(y)}: {v:.2f} {unit}")
    
    # Graf predikcí
    fig, ax = plt.subplots()
    ax.plot(df["year"], df[column], label="Historie", color=color)
    ax.plot(years_pred, values_pred, "ro", label="Predikce")
    ax.set_title(f"Predikce {name} v Brně")
    ax.set_xlabel("Rok")
    ax.set_ylabel(unit)
    ax.legend()
    st.pyplot(fig)
    
    # Shrnutí
    st.markdown(f"""
    Model předpokládá lineární vývoj {name.lower()}:  
    **y = {coef[0]:.4f} · rok + {coef[1]:.2f}**
    
    To znamená, že {name.lower()} roste v průměru o **{coef[0]:.2f} {unit} za rok**.  
    Predikce na 1000 let jsou velmi nejisté a slouží spíše jako ilustrace trendu než přesná prognóza.
    """)
