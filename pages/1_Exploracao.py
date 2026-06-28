import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px

from utils_wb import (
    carregar_base_worldbank,
    filtrar_base,
    formatar_numero,
    ROTULOS_INDICADOR,
)

st.set_page_config(page_title="Exploração", page_icon="📊", layout="wide")
st.title("📊 Exploração dos dados")

with st.spinner("Carregando dados..."):
    df = carregar_base_worldbank()

anos = sorted(df["ano"].dropna().unique())
regioes_disponiveis = sorted(df["regiao"].dropna().unique())
rendas_disponiveis = sorted(df["grupo_renda"].dropna().unique())

st.sidebar.header("Filtros")
ano = st.sidebar.slider("Ano", int(min(anos)), int(max(anos)), int(max(anos)))
regioes = st.sidebar.multiselect("Regiões", regioes_disponiveis, default=[])
rendas = st.sidebar.multiselect("Grupos de renda", rendas_disponiveis, default=[])

indicador_mapa = st.sidebar.selectbox(
    "Indicador do mapa",
    options=list(ROTULOS_INDICADOR.keys()),
    format_func=lambda x: ROTULOS_INDICADOR[x],
)

df_ano = filtrar_base(df, ano, regioes, rendas)

# --------------------------------------------------------------------------
# Cards
# --------------------------------------------------------------------------
df_cards = df_ano.dropna(subset=["co2_pc"])
col1, col2, col3, col4 = st.columns(4)
col1.metric("Países no filtro", df_ano["pais"].nunique())
col2.metric("CO₂ médio per capita", formatar_numero(df_cards["co2_pc"].mean()))
col3.metric("PIB per capita médio", f"US$ {formatar_numero(df_ano['pib_pc'].mean())}")
col4.metric("Energia renovável média", f"{formatar_numero(df_ano['renovaveis'].mean())}%")

if df_ano.empty:
    st.warning("Nenhum país no filtro selecionado. Ajuste os filtros na barra lateral.")
    st.stop()

# --------------------------------------------------------------------------
# 1. Mapa
# --------------------------------------------------------------------------
st.subheader("1. Mapa mundial")
df_mapa = df_ano.dropna(subset=[indicador_mapa]).copy()
fig_mapa = px.choropleth(
    df_mapa,
    locations="iso3",
    color=indicador_mapa,
    hover_name="pais",
    color_continuous_scale="Viridis",
    labels={indicador_mapa: ROTULOS_INDICADOR[indicador_mapa]},
    title=f"{ROTULOS_INDICADOR[indicador_mapa]} por país — {ano}",
)
fig_mapa.update_layout(margin=dict(l=0, r=0, t=40, b=0))
st.plotly_chart(fig_mapa, use_container_width=True)

# --------------------------------------------------------------------------
# 2. Scatter renda x emissões (com opção de log e linha de tendência)
# --------------------------------------------------------------------------
st.subheader("2. Relação entre renda e emissões")
c1, c2 = st.columns(2)
escala_log = c1.toggle("Escala log no eixo do PIB", value=True,
                       help="A relação CO₂ × renda fica mais clara em escala logarítmica.")
mostrar_tendencia = c2.toggle("Mostrar linha de tendência", value=True)

df_scatter = df_ano.dropna(subset=["pib_pc", "co2_pc", "regiao"]).copy()
x_enc = alt.X("pib_pc:Q", title="PIB per capita real (US$)",
              scale=alt.Scale(type="log") if escala_log else alt.Scale(type="linear"))

base_scatter = alt.Chart(df_scatter).encode(
    x=x_enc,
    y=alt.Y("co2_pc:Q", title="CO₂ per capita (t/hab)"),
)
pontos = base_scatter.mark_circle(size=70, opacity=0.75).encode(
    color=alt.Color("regiao:N", title="Região"),
    tooltip=["pais", "regiao", "grupo_renda", "pib_pc", "co2_pc", "urbanizacao", "renovaveis"],
)
grafico = pontos
if mostrar_tendencia and len(df_scatter) > 2:
    tendencia = base_scatter.transform_regression(
        "pib_pc", "co2_pc",
        method=("log" if escala_log else "linear"),
    ).mark_line(color="black", strokeDash=[5, 3])
    grafico = pontos + tendencia

st.altair_chart(grafico.properties(height=450).interactive(), use_container_width=True)

# --------------------------------------------------------------------------
# 3. Ranking
# --------------------------------------------------------------------------
st.subheader("3. Ranking de emissões per capita")
top_n = st.slider("Número de países no ranking", 5, 30, 15)
df_ranking = df_ano.dropna(subset=["co2_pc"]).sort_values("co2_pc", ascending=False).head(top_n)
ranking = (
    alt.Chart(df_ranking)
    .mark_bar()
    .encode(
        x=alt.X("co2_pc:Q", title="CO₂ per capita (t/hab)"),
        y=alt.Y("pais:N", sort="-x", title="País"),
        color=alt.Color("co2_pc:Q", scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["pais", "co2_pc", "pib_pc", "regiao", "grupo_renda"],
    )
    .properties(height=max(300, top_n * 24))
)
st.altair_chart(ranking, use_container_width=True)

# --------------------------------------------------------------------------
# 4. Heatmap de correlação entre indicadores
# --------------------------------------------------------------------------
st.subheader("4. Correlação entre indicadores")
cols_corr = ["co2_pc", "pib_pc", "urbanizacao", "eletricidade", "renovaveis"]
corr = df_ano[cols_corr].corr().stack().reset_index()
corr.columns = ["var_x", "var_y", "correl"]
corr["var_x"] = corr["var_x"].map(ROTULOS_INDICADOR)
corr["var_y"] = corr["var_y"].map(ROTULOS_INDICADOR)

heat = (
    alt.Chart(corr)
    .mark_rect()
    .encode(
        x=alt.X("var_x:N", title=None),
        y=alt.Y("var_y:N", title=None),
        color=alt.Color("correl:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1]),
                        title="Correlação"),
        tooltip=["var_x", "var_y", alt.Tooltip("correl:Q", format=".2f")],
    )
    .properties(height=320)
)
texto = heat.mark_text(baseline="middle").encode(
    text=alt.Text("correl:Q", format=".2f"),
    color=alt.condition("abs(datum.correl) > 0.5", alt.value("white"), alt.value("black")),
)
st.altair_chart(heat + texto, use_container_width=True)

with st.expander("Ver dados do ano selecionado"):
    st.dataframe(df_ano, use_container_width=True)
