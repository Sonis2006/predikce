import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 
st.set_page_config(page_title="Klimatická predikce Brno", layout="wide")
st.title("🌡️ Historický vývoj a predikce ročního průměru teplot v Brně")
 
# === 1. Načtení dat ===
@st.cache_data
def load_temperature_data():
    years = list(range(1900, 2024))
    temperature = np.linspace(8.0, 10.5, len(years)) + np.random.normal(0, 0.2, len(years))
    return pd.DataFrame({"year": years, "temperature": temperature})
 
df = load_temperature_data()
 
# === 2. Zobrazení historických dat ===
st.subheader("📊 Historický roční průměr teplot v Brně")
 
fig1, ax1 = plt.subplots()
ax1.plot(df["year"], df["temperature"], color="blue")
ax1.set_title("Roční průměr teplot v Brně")
ax1.set_xlabel("Rok")
ax1.set_ylabel("Roční průměr teplot (°C)")
st.pyplot(fig1)
 
# === 3. Predikce pomocí numpy.polyfit ===
def predict_future(df, column, horizons):
    x = df["year"].values
    y = df[column].values
    coef = np.polyfit(x, y, 1)
    future_years = np.array([x.max() + h for h in horizons])
    predictions = coef[0] * future_years + coef[1]
    return future_years, predictions, coef
 
horizons = [10, 100, 1000]
years_pred, values_pred, coef = predict_future(df, "temperature", horizons)
 
# === 4. Výstup predikcí ===
st.subheader("📈 Predikce ročního průměru teplot")
for y, v in zip(years_pred, values_pred):
    st.write(f"Rok {int(y)}: {v:.2f} °C")
 
fig2, ax2 = plt.subplots()
ax2.plot(df["year"], df["temperature"], label="Historie", color="blue")
ax2.plot(years_pred, values_pred, "ro", label="Predikce")
ax2.set_title("Predikce ročního průměru teplot v Brně")
ax2.set_xlabel("Rok")
ax2.set_ylabel("Roční průměr teplot (°C)")
ax2.legend()
st.pyplot(fig2)
 
# === 5. Shrnutí ===
st.subheader("🧠 Shrnutí")
st.markdown(f"""
Model předpokládá lineární vývoj ročního průměru teplot:  
**y = {coef[0]:.4f} · rok + {coef[1]:.2f}**
 
To znamená, že roční průměr teplot roste v průměru o **{coef[0]:.2f} °C za rok**.  
Predikce na 1000 let jsou velmi nejisté a slouží spíše jako ilustrace trendu než přesná prognóza.
""")
