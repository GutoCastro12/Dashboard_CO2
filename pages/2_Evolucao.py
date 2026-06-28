import streamlit as st
import pandas as pd
import altair as alt

from utils_wb import carregar_base_worldbank, ROTULOS_INDICADOR, formatar_numero

st.set_page_config(page_title="Evolução temporal", page_icon="📈", layout="wide")
st.title("📈 Evolução temporal")

with st.spinner("Carregando dados..."):
    df = carregar_base_worldbank()

paises_padrao = [p for p in ["Brazil", "United States", "China", "Germany", "India"]
                 if p in df["pais"].unique()]
paises = st.multiselect(
    "Escolha países",
    options=sorted(df["pais"].dropna().unique()),
    default=paises_padrao,
)

c1, c2, c3 = st.columns(3)
indicador = c1.selectbox(
    "Indicador",
    options=list(ROTULOS_INDICADOR.keys()),
    format_func=lambda x: ROTULOS_INDICADOR[x],
)
janela = c2.slider("Média móvel (anos)", 1, 10, 3)
indexar = c3.toggle("Indexar para 100 no 1º ano",
                    help="Compara o crescimento relativo de cada país a partir do início da série.")

if not paises:
    st.warning("Selecione pelo menos um país.")
    st.stop()

df_linha = df[df["pais"].isin(paises)].copy().sort_values(["pais", "ano"])
df_linha = df_linha.dropna(subset=[indicador])
df_linha["media_movel"] = (
    df_linha.groupby("pais")[indicador]
    .transform(lambda s: s.rolling(janela, min_periods=1).mean())
)

# Indexação base 100
coluna_valor = indicador
titulo_y = ROTULOS_INDICADOR[indicador]
if indexar:
    def _index(s):
        base = s.dropna().iloc[0] if s.dropna().size else None
        return s / base * 100 if base not in (None, 0) else s
    df_linha["indexado"] = df_linha.groupby("pais")[indicador].transform(_index)
    coluna_valor = "indexado"
    titulo_y = f"{ROTULOS_INDICADOR[indicador]} (base 100 = 1º ano)"

# --------------------------------------------------------------------------
# Cards: variação no período por país
# --------------------------------------------------------------------------
st.subheader("Variação no período")
cards = st.columns(min(len(paises), 5))
for i, pais in enumerate(paises[:5]):
    sub = df_linha[df_linha["pais"] == pais].dropna(subset=[indicador])
    if len(sub) >= 2:
        ini, fim = sub.iloc[0][indicador], sub.iloc[-1][indicador]
        var = (fim / ini - 1) * 100 if ini not in (0, None) else float("nan")
        cards[i].metric(pais, formatar_numero(fim), f"{formatar_numero(var, 1)}%")

# --------------------------------------------------------------------------
# 1. Série
# --------------------------------------------------------------------------
st.subheader("1. Evolução dos países selecionados")
linha = (
    alt.Chart(df_linha)
    .mark_line(point=True)
    .encode(
        x=alt.X("ano:O", title="Ano"),
        y=alt.Y(f"{coluna_valor}:Q", title=titulo_y),
        color=alt.Color("pais:N", title="País"),
        tooltip=["pais", "ano", alt.Tooltip(f"{coluna_valor}:Q", format=".2f"),
                 "regiao", "grupo_renda"],
    )
    .properties(height=420)
    .interactive()
)
st.altair_chart(linha, use_container_width=True)

# --------------------------------------------------------------------------
# 2. Média móvel
# --------------------------------------------------------------------------
st.subheader(f"2. Série suavizada (média móvel de {janela} anos)")
linha_mm = (
    alt.Chart(df_linha.dropna(subset=["media_movel"]))
    .mark_line()
    .encode(
        x=alt.X("ano:O", title="Ano"),
        y=alt.Y("media_movel:Q", title=f"Média móvel de {ROTULOS_INDICADOR[indicador]}"),
        color=alt.Color("pais:N", title="País"),
        tooltip=["pais", "ano", alt.Tooltip("media_movel:Q", format=".2f")],
    )
    .properties(height=420)
)
st.altair_chart(linha_mm, use_container_width=True)

with st.expander("Ver dados dos países selecionados"):
    st.dataframe(df_linha, use_container_width=True)
