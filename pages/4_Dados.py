import streamlit as st
import pandas as pd

from utils_wb import carregar_base_worldbank, baixar_csv, ROTULOS_INDICADOR, formatar_numero

st.set_page_config(page_title="Dados", page_icon="🗂️", layout="wide")
st.title("🗂️ Base de dados")

with st.spinner("Carregando dados..."):
    df = carregar_base_worldbank()

st.markdown(
    """
    Base consolidada após o *merge* dos indicadores do World Bank com os metadados de
    países. A unidade é **país-ano**. Use os filtros para explorar e baixe o CSV para
    reprodução.
    """
)

# --------------------------------------------------------------------------
# Filtros
# --------------------------------------------------------------------------
c1, c2, c3 = st.columns(3)
anos = sorted(df["ano"].dropna().unique())
faixa = c1.select_slider("Faixa de anos", options=[int(a) for a in anos],
                         value=(int(min(anos)), int(max(anos))))
regioes = c2.multiselect("Regiões", sorted(df["regiao"].dropna().unique()))
busca = c3.text_input("Buscar país", "")

df_f = df[(df["ano"] >= faixa[0]) & (df["ano"] <= faixa[1])].copy()
if regioes:
    df_f = df_f[df_f["regiao"].isin(regioes)]
if busca.strip():
    df_f = df_f[df_f["pais"].str.contains(busca.strip(), case=False, na=False)]

c1, c2, c3 = st.columns(3)
c1.metric("Linhas no filtro", f"{len(df_f):,}".replace(",", "."))
c2.metric("Países", df_f["pais"].nunique())
c3.metric("Anos", df_f["ano"].nunique())

# --------------------------------------------------------------------------
# Cobertura / valores faltantes por indicador
# --------------------------------------------------------------------------
st.subheader("Completude dos indicadores (no filtro)")
indicadores = list(ROTULOS_INDICADOR.keys())
linhas = []
total = len(df_f)
for ind in indicadores:
    preenchidos = int(df_f[ind].notna().sum())
    pct = preenchidos / total * 100 if total else 0
    linhas.append({
        "Indicador": ROTULOS_INDICADOR[ind],
        "Preenchidos": preenchidos,
        "Faltantes": total - preenchidos,
        "% preenchido": round(pct, 1),
    })
st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

# --------------------------------------------------------------------------
# Download + tabela
# --------------------------------------------------------------------------
st.download_button(
    "Baixar dados filtrados em CSV",
    data=baixar_csv(df_f),
    file_name="base_worldbank_filtrada.csv",
    mime="text/csv",
)
st.dataframe(df_f, use_container_width=True)
