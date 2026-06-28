import streamlit as st
import pandas as pd
import altair as alt

from utils_wb import (
    carregar_base_worldbank,
    formatar_numero,
    baixar_csv,
    INDICADORES,
)

st.set_page_config(
    page_title="CO₂ e Desenvolvimento",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Desenvolvimento econômico e emissões de CO₂")
st.caption("Dashboard interativo com dados abertos do World Bank · unidade de observação: país-ano")

st.markdown(
    """
    Este painel investiga **o que explica as diferenças de emissões de CO₂ per capita
    entre países** — combinando renda, urbanização, acesso à eletricidade e estrutura
    energética. Os dados vêm direto da API pública do **World Bank** e são consolidados
    por *merge* usando as chaves `iso3` e `ano`.
    """
)

with st.spinner("Carregando dados do World Bank..."):
    df = carregar_base_worldbank()

# --------------------------------------------------------------------------
# Cards de resumo
# --------------------------------------------------------------------------
anos = sorted(df["ano"].dropna().unique())
anos_com_co2 = df.dropna(subset=["co2_pc"]).groupby("ano")["pais"].count()
ultimo_ano_bom = int(anos_com_co2[anos_com_co2 >= 100].index.max())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Países na base", df["pais"].nunique())
col2.metric("Período", f"{int(min(anos))}–{int(max(anos))}")
col3.metric("Último ano com boa cobertura", ultimo_ano_bom)
col4.metric("Observações (país-ano)", f"{len(df):,}".replace(",", "."))

st.markdown("---")

# --------------------------------------------------------------------------
# Achados rápidos calculados a partir dos dados
# --------------------------------------------------------------------------
st.header("📌 Panorama do último ano com boa cobertura")

df_ano = df[(df["ano"] == ultimo_ano_bom)].dropna(subset=["co2_pc"]).copy()
correl = df.dropna(subset=["co2_pc", "pib_pc"])[["co2_pc", "pib_pc"]].corr().iloc[0, 1]

if not df_ano.empty:
    top = df_ano.sort_values("co2_pc", ascending=False).iloc[0]
    bottom = df_ano.sort_values("co2_pc").iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric(f"Maior emissor ({ultimo_ano_bom})", top["pais"],
              f'{formatar_numero(top["co2_pc"])} t/hab')
    c2.metric("Menor emissor", bottom["pais"],
              f'{formatar_numero(bottom["co2_pc"])} t/hab')
    c3.metric("Correlação CO₂ × PIB per capita", formatar_numero(correl, 2),
              help="Correlação de Pearson em toda a base (todos os anos).")

st.markdown(
    f"""
    No conjunto consolidado, CO₂ per capita e PIB per capita têm correlação positiva
    de **{formatar_numero(correl, 2)}**, mas a relação não é linear: ela tende a se
    achatar nos países mais ricos — exatamente a hipótese que o modelo da aba
    **🤖 Modelo** testa com o termo quadrático do log do PIB (curva de Kuznets ambiental).
    """
)

# Distribuição de CO2 por grupo de renda no último ano bom
st.subheader("CO₂ per capita por grupo de renda")
df_box = df_ano.dropna(subset=["grupo_renda"])
if not df_box.empty:
    box = (
        alt.Chart(df_box)
        .mark_boxplot(extent="min-max")
        .encode(
            x=alt.X("grupo_renda:N", title="Grupo de renda", sort="-y"),
            y=alt.Y("co2_pc:Q", title="CO₂ per capita (t/hab)"),
            color=alt.Color("grupo_renda:N", legend=None),
            tooltip=["pais", "co2_pc", "grupo_renda"],
        )
        .properties(height=360)
    )
    st.altair_chart(box, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------
# Pergunta do projeto e bases
# --------------------------------------------------------------------------
st.header("❓ Pergunta do projeto")
st.markdown(
    """
    > **Quais características econômicas e energéticas ajudam a explicar as diferenças
    de emissões de CO₂ per capita entre países?**

    A variável-alvo é **CO₂ per capita**. As variáveis explicativas são PIB per capita,
    urbanização, acesso à eletricidade e participação de energia renovável no consumo
    final de energia.
    """
)

st.header("🧩 Bases utilizadas (merge/join)")
linhas = [
    {"Coluna na base": coluna, "Indicador": info["nome"], "Código World Bank": info["codigo"]}
    for coluna, info in INDICADORES.items()
]
st.dataframe(pd.DataFrame(linhas), use_container_width=True, hide_index=True)

st.markdown(
    """
    Cada indicador é baixado de um **endpoint separado** da API e vira uma base própria.
    As bases são unidas por `iso3` + `ano`. Em seguida, faz-se um *join* com uma base
    **auxiliar de metadados** do World Bank para acrescentar **região** e **grupo de
    renda**. Ou seja, a base final é a união de várias fontes distintas.
    """
)

with st.expander("👀 Ver prévia da base consolidada e baixar CSV"):
    st.dataframe(df.head(50), use_container_width=True)
    st.download_button(
        "Baixar base consolidada em CSV",
        data=baixar_csv(df),
        file_name="base_worldbank_completa.csv",
        mime="text/csv",
    )

st.info(
    "Use o menu lateral para navegar: **📊 Exploração**, **📈 Evolução temporal**, "
    "**🤖 Modelo** e **🗂️ Dados**."
)
