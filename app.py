import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Dashboard Meteorológico - São Luiz do Paraitinga", page_icon="🌤️", layout="wide")
st.title("🌤️ Dashboard Meteorológico - São Luiz do Paraitinga - SP")
st.caption("Fonte: INMET - Estações Automáticas")
st.markdown("---")

DATA_PATH = os.getenv("DATA_PATH", "data/inmet_data_sao_luiz_do_paraitinga_combined.csv")

@st.cache_data(show_spinner=False)
def load_data(path: str):
    try:
        df = pd.read_csv(path, parse_dates=["DATA"])
        for col in ["TEMPERATURA_MEDIA","TEMPERATURA_MAXIMA","TEMPERATURA_MINIMA","UMIDADE_RELATIVA",
                    "PRECIPITACAO","VELOCIDADE_VENTO","PRESSAO_ATMOSFERICA"]:
            if col not in df.columns:
                df[col] = np.nan
        return df.sort_values("DATA")
    except Exception as e:
        st.error(f"Não foi possível carregar {path}: {e}")
        return pd.DataFrame(columns=["DATA"])

df = load_data(DATA_PATH)

if df.empty:
    st.warning("Base ainda não disponível. Rode o script de coleta (`fetch_inmet.py`) ou aguarde o GitHub Actions atualizar.")
    st.stop()

st.sidebar.header("🔧 Filtros")
min_date, max_date = df["DATA"].min().date(), df["DATA"].max().date()
rng = st.sidebar.date_input("Período", value=(min_date, max_date), min_value=min_date, max_value=max_date)
if isinstance(rng, (list, tuple)) and len(rng) == 2:
    start, end = rng
    dff = df[(df["DATA"].dt.date >= start) & (df["DATA"].dt.date <= end)].copy()
else:
    dff = df.copy()

st.header("📊 Métricas Principais")
c1,c2,c3,c4 = st.columns(4)
with c1:
    val = dff["TEMPERATURA_MEDIA"].mean()
    st.metric("Temperatura média", f"{val:.1f} °C" if pd.notna(val) else "—")
with c2:
    val = dff["UMIDADE_RELATIVA"].mean()
    st.metric("Umidade média", f"{val:.1f} %" if pd.notna(val) else "—")
with c3:
    val = dff["PRECIPITACAO"].sum()
    st.metric("Precipitação total", f"{val:.1f} mm" if pd.notna(val) else "—")
with c4:
    val = dff["VELOCIDADE_VENTO"].mean()
    st.metric("Velocidade do vento", f"{val:.1f} m/s" if pd.notna(val) else "—")

st.markdown("---")
st.header("📈 Análise Temporal")

fig_temp = px.line(dff, x="DATA", y=["TEMPERATURA_MAXIMA","TEMPERATURA_MINIMA","TEMPERATURA_MEDIA"],
                   labels={"value":"Temperatura (°C)", "variable":"Série"}, title="Temperaturas ao longo do tempo")
st.plotly_chart(fig_temp, use_container_width=True)

fig_p = px.bar(dff, x="DATA", y="PRECIPITACAO", title="Precipitação diária (mm)")
st.plotly_chart(fig_p, use_container_width=True)

fig_u = px.line(dff, x="DATA", y="UMIDADE_RELATIVA", title="Umidade relativa (%)")
st.plotly_chart(fig_u, use_container_width=True)

st.markdown("---")
st.header("🤖 Aprendizado de Máquina (exemplo)")
st.caption("Regressão Linear para estimar Temperatura Média usando mês e dia do ano.")

if dff["TEMPERATURA_MEDIA"].notna().sum() > 30:
    df_ml = dff[["DATA","TEMPERATURA_MEDIA"]].dropna().copy()
    df_ml["MES"] = df_ml["DATA"].dt.month
    df_ml["DIA_DO_ANO"] = df_ml["DATA"].dt.dayofyear
    X = df_ml[["MES","DIA_DO_ANO"]]
    y = df_ml["TEMPERATURA_MEDIA"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    c1, c2 = st.columns(2)
    with c1:
        st.metric("MSE", f"{mse:.3f}")
        st.metric("R²", f"{r2:.3f}")
    with c2:
        fig_ml = go.Figure()
        fig_ml.add_trace(go.Scatter(x=y_test, y=y_pred, mode="markers", name="Predições"))
        lo = float(min(y_test.min(), y_pred.min()))
        hi = float(max(y_test.max(), y_pred.max()))
        fig_ml.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines", name="Perfeito", line=dict(dash="dash")))
        fig_ml.update_layout(title="Predição vs. Real (Temp. Média)",
                             xaxis_title="Real (°C)", yaxis_title="Predito (°C)")
        st.plotly_chart(fig_ml, use_container_width=True)
else:
    st.info("Ainda não há dados suficientes de temperatura média para treinar o modelo.")

st.markdown("---")
st.header("📄 Dados brutos")
if st.checkbox("Mostrar dados filtrados"):
    st.dataframe(dff)
st.download_button("📥 Baixar CSV filtrado", data=dff.to_csv(index=False).encode("utf-8"),
                   file_name="dados_filtrados.csv", mime="text/csv")
