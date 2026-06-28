import streamlit as st
import numpy as np
import pandas as pd
import altair as alt

from utils_wb import (
    carregar_base_worldbank,
    criar_base_modelo,
    separar_treino_teste,
    estimar_modelo,
    validacao_cruzada,
    NOMES_VARIAVEIS,
    formatar_numero,
    estrelas_significancia,
)

st.set_page_config(page_title="Modelo", page_icon="🤖", layout="wide")
st.title("🤖 Modelo de regressão interativo")

st.markdown(
    """
    O objetivo é **prever CO₂ per capita** a partir de variáveis econômicas, urbanas e
    energéticas. O estimador é uma **regressão por mínimos quadrados (MQO)** implementada
    com `numpy`, agora com **inferência completa** (erros-padrão, estatísticas *t* e
    *p*-valores), **validação cruzada**, comparação com um **modelo trivial** e
    **diagnóstico de resíduos**.
    """
)

with st.spinner("Carregando dados..."):
    df = carregar_base_worldbank()

# Anos com cobertura suficiente
colunas_base = ["co2_pc", "log_pib_pc", "urbanizacao", "eletricidade", "renovaveis"]
cobertura = df.dropna(subset=colunas_base).groupby("ano")["pais"].count()
anos_validos = sorted(int(a) for a in cobertura[cobertura >= 50].index)

regioes_disponiveis = sorted(df["regiao"].dropna().unique())
rendas_disponiveis = sorted(df["grupo_renda"].dropna().unique())

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
st.sidebar.header("Configuração do modelo")
ano = st.sidebar.select_slider("Ano da regressão", options=anos_validos, value=max(anos_validos))
st.sidebar.caption(f"Apenas anos com ≥ 50 países completos. Último disponível: {max(anos_validos)}.")

regioes = st.sidebar.multiselect("Regiões", regioes_disponiveis, default=[])
rendas = st.sidebar.multiselect("Grupos de renda", rendas_disponiveis, default=[])

st.sidebar.markdown("**Especificação**")
usar_log = st.sidebar.checkbox("Usar log do PIB per capita", value=True)
usar_quadratico = st.sidebar.checkbox("Incluir termo quadrático (log PIB²)", value=True,
                                      help="Permite testar a curva de Kuznets ambiental.")

variaveis_disponiveis = []
if usar_log:
    variaveis_disponiveis.append("log_pib_pc")
    if usar_quadratico:
        variaveis_disponiveis.append("log_pib_pc_2")
else:
    variaveis_disponiveis.append("pib_pc")
variaveis_disponiveis += ["urbanizacao", "eletricidade", "renovaveis"]

variaveis_x = st.sidebar.multiselect(
    "Variáveis explicativas",
    options=variaveis_disponiveis,
    default=variaveis_disponiveis,
    format_func=lambda x: NOMES_VARIAVEIS.get(x, x),
)

st.sidebar.markdown("**Estimação e avaliação**")
metodo = st.sidebar.radio("Método", ["ols", "ridge"],
                          format_func=lambda x: {"ols": "MQO (OLS)", "ridge": "Ridge (L2)"}[x])
alpha = 0.0
if metodo == "ridge":
    alpha = st.sidebar.slider("Regularização α (Ridge)", 0.0, 50.0, 5.0, 0.5)

prop_treino = st.sidebar.slider("Proporção para treino", 0.50, 0.90, 0.70, 0.05)
n_folds = st.sidebar.slider("Folds da validação cruzada", 3, 10, 5)
seed = st.sidebar.number_input("Semente aleatória", 1, 9999, 42, 1)

if len(variaveis_x) == 0:
    st.warning("Escolha pelo menos uma variável explicativa.")
    st.stop()

# --------------------------------------------------------------------------
# Base do modelo
# --------------------------------------------------------------------------
df_reg = criar_base_modelo(df, ano, variaveis_x, regioes, rendas)

st.subheader("Base usada no modelo")
c1, c2, c3 = st.columns(3)
c1.metric("Ano", ano)
c2.metric("Países disponíveis", len(df_reg))
c3.metric("Variáveis explicativas", len(variaveis_x))

if len(df_reg) < max(25, len(variaveis_x) + 10):
    st.error("Amostra pequena demais para estimar com segurança. Amplie os filtros ou troque o ano.")
    st.dataframe(df_reg, use_container_width=True)
    st.stop()

df_treino, df_teste = separar_treino_teste(df_reg, prop_treino=prop_treino, seed=int(seed))
res = estimar_modelo(df_treino, df_teste, variaveis_x, metodo=metodo, alpha=alpha)

m_tr, m_te, m_base = res["metricas_treino"], res["metricas_teste"], res["metricas_baseline"]

# --------------------------------------------------------------------------
# Avaliação no teste + comparação com baseline
# --------------------------------------------------------------------------
st.subheader("Avaliação no conjunto de teste")
c1, c2, c3, c4 = st.columns(4)
c1.metric("R² (teste)", formatar_numero(m_te["R²"], 3))
c2.metric("RMSE (teste)", formatar_numero(m_te["RMSE"]),
          delta=f'{formatar_numero(m_te["RMSE"] - m_base["RMSE"])} vs baseline',
          delta_color="inverse")
c3.metric("MAE (teste)", formatar_numero(m_te["MAE"]))
c4.metric("Obs. teste", len(df_teste))

ganho = (1 - m_te["RMSE"] / m_base["RMSE"]) * 100 if m_base["RMSE"] else float("nan")
st.caption(
    f"O **baseline** (prever sempre a média do treino) tem RMSE de "
    f"{formatar_numero(m_base['RMSE'])}. O modelo reduz o erro em "
    f"**{formatar_numero(ganho, 1)}%** — é assim que se sabe que ele aprendeu algo útil."
)

# Sobreajuste: treino vs teste
gap = m_tr["R²"] - m_te["R²"]
tabela_metricas = pd.DataFrame([
    {"base": "treino", **m_tr},
    {"base": "teste", **m_te},
    {"base": "baseline (teste)", **m_base},
])
st.dataframe(tabela_metricas.round(3), use_container_width=True, hide_index=True)
if gap > 0.15:
    st.warning(
        f"R² cai {formatar_numero(gap, 2)} do treino para o teste — possível **sobreajuste**. "
        "Tente reduzir variáveis, usar Ridge ou aumentar a amostra."
    )

# --------------------------------------------------------------------------
# Validação cruzada k-fold
# --------------------------------------------------------------------------
st.subheader(f"Validação cruzada ({n_folds}-fold)")
cv = validacao_cruzada(df_reg, variaveis_x, k=int(n_folds), seed=int(seed),
                       metodo=metodo, alpha=alpha)
cc1, cc2, cc3 = st.columns(3)
cc1.metric("R² médio (CV)", formatar_numero(cv["R²"].mean(), 3),
           f'± {formatar_numero(cv["R²"].std(), 3)}')
cc2.metric("RMSE médio (CV)", formatar_numero(cv["RMSE"].mean()))
cc3.metric("MAE médio (CV)", formatar_numero(cv["MAE"].mean()))

cv_plot = cv.copy()
barras_cv = (
    alt.Chart(cv_plot)
    .mark_bar()
    .encode(
        x=alt.X("fold:O", title="Fold"),
        y=alt.Y("R²:Q", title="R² no fold"),
        color=alt.Color("R²:Q", scale=alt.Scale(scheme="greens"), legend=None),
        tooltip=["fold", alt.Tooltip("R²:Q", format=".3f"),
                 alt.Tooltip("RMSE:Q", format=".3f"), "n_teste"],
    )
    .properties(height=240)
)
regra_media = alt.Chart(pd.DataFrame({"y": [cv["R²"].mean()]})).mark_rule(
    strokeDash=[5, 3], color="black").encode(y="y:Q")
st.altair_chart(barras_cv + regra_media, use_container_width=True)
st.caption("A validação cruzada mostra se o desempenho é estável entre diferentes divisões dos dados.")

# --------------------------------------------------------------------------
# Coeficientes com inferência
# --------------------------------------------------------------------------
st.subheader("Coeficientes estimados")
nomes = ["constante"] + variaveis_x
beta = res["beta"]
aj = res["ajuste"]

dados_coef = {"variável": [NOMES_VARIAVEIS.get(v, v) for v in nomes], "coeficiente": beta}
if metodo == "ols" and aj["se"] is not None:
    dados_coef["erro-padrão"] = aj["se"]
    dados_coef["estat. t"] = aj["t"]
    dados_coef["p-valor"] = aj["p"]
    dados_coef["signif."] = [estrelas_significancia(p) for p in aj["p"]]
coef = pd.DataFrame(dados_coef)
st.dataframe(coef.round(4), use_container_width=True, hide_index=True)
if metodo == "ols":
    st.caption("Significância: *** p<0,01 · ** p<0,05 · * p<0,10. "
               f"R² ajustado no treino: **{formatar_numero(aj['r2_ajustado'], 3)}**.")
else:
    st.caption("No Ridge os coeficientes são encolhidos para reduzir variância; "
               "não calculamos p-valores nesse caso.")

# Importância (coeficientes padronizados)
res_pad = estimar_modelo(df_treino, df_teste, variaveis_x, metodo=metodo, alpha=alpha, padronizar=True)
imp = pd.DataFrame({
    "variável": [NOMES_VARIAVEIS.get(v, v) for v in variaveis_x],
    "efeito_padronizado": res_pad["beta"][1:],
})
imp["sentido"] = np.where(imp["efeito_padronizado"] >= 0, "positivo", "negativo")
st.markdown("**Importância relativa (coeficientes padronizados)**")
barras_imp = (
    alt.Chart(imp)
    .mark_bar()
    .encode(
        x=alt.X("efeito_padronizado:Q", title="Efeito sobre CO₂ (em desvios-padrão)"),
        y=alt.Y("variável:N", sort="-x", title=None),
        color=alt.Color("sentido:N", scale=alt.Scale(
            domain=["positivo", "negativo"], range=["#2c7fb8", "#d95f0e"]), title="Sentido"),
        tooltip=["variável", alt.Tooltip("efeito_padronizado:Q", format=".3f")],
    )
    .properties(height=max(180, len(variaveis_x) * 45))
)
st.altair_chart(barras_imp, use_container_width=True)
st.caption("Com X padronizado, os coeficientes ficam comparáveis: barras maiores = maior peso na previsão.")

# --------------------------------------------------------------------------
# Curva de Kuznets ambiental (se houver termo quadrático)
# --------------------------------------------------------------------------
if "log_pib_pc" in variaveis_x and "log_pib_pc_2" in variaveis_x:
    idx_lin = nomes.index("log_pib_pc")
    idx_quad = nomes.index("log_pib_pc_2")
    b1, b2 = beta[idx_lin], beta[idx_quad]
    st.subheader("📐 Curva de Kuznets ambiental")
    if b2 < 0 and b1 > 0:
        log_pib_otimo = -b1 / (2 * b2)
        pib_otimo = float(np.exp(log_pib_otimo))
        st.success(
            f"O sinal dos coeficientes (log PIB **+**, log PIB² **−**) é consistente com a "
            f"**hipótese de Kuznets**: emissões sobem com a renda e depois desaceleram. "
            f"O ponto de virada estimado fica em torno de **US$ {formatar_numero(pib_otimo, 0)}** "
            f"de PIB per capita."
        )
    else:
        st.info("Os coeficientes do log do PIB não formam o U-invertido típico da curva de "
                "Kuznets neste recorte. Experimente outro ano ou conjunto de filtros.")

# --------------------------------------------------------------------------
# CO2 observado vs previsto
# --------------------------------------------------------------------------
st.subheader("CO₂ observado vs. previsto (teste)")
df_plot = df_teste.copy()
df_plot["co2_previsto"] = res["pred_teste"]
df_plot["residuo"] = df_plot["co2_pc"] - df_plot["co2_previsto"]

disp = (
    alt.Chart(df_plot)
    .mark_circle(size=80, opacity=0.75)
    .encode(
        x=alt.X("co2_pc:Q", title="CO₂ observado"),
        y=alt.Y("co2_previsto:Q", title="CO₂ previsto"),
        color=alt.Color("regiao:N", title="Região"),
        tooltip=["pais", "regiao", "grupo_renda",
                 alt.Tooltip("co2_pc:Q", format=".2f"),
                 alt.Tooltip("co2_previsto:Q", format=".2f")],
    )
    .properties(height=440)
    .interactive()
)
lim = [float(min(df_plot["co2_pc"].min(), df_plot["co2_previsto"].min())),
       float(max(df_plot["co2_pc"].max(), df_plot["co2_previsto"].max()))]
ref = alt.Chart(pd.DataFrame({"x": lim, "y": lim})).mark_line(strokeDash=[6, 4]).encode(x="x", y="y")
st.altair_chart(disp + ref, use_container_width=True)
st.caption("Pontos sobre a linha pontilhada = previsão perfeita. Acima/abaixo = erro do modelo.")

# --------------------------------------------------------------------------
# Diagnóstico de resíduos
# --------------------------------------------------------------------------
st.subheader("Diagnóstico de resíduos")
d1, d2 = st.columns(2)
res_vs_fit = (
    alt.Chart(df_plot)
    .mark_circle(size=60, opacity=0.7)
    .encode(
        x=alt.X("co2_previsto:Q", title="CO₂ previsto"),
        y=alt.Y("residuo:Q", title="Resíduo (obs − prev)"),
        tooltip=["pais", alt.Tooltip("residuo:Q", format=".2f")],
    )
    .properties(height=300)
)
zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="red", strokeDash=[4, 4]).encode(y="y:Q")
d1.markdown("**Resíduos vs. valores previstos**")
d1.altair_chart(res_vs_fit + zero, use_container_width=True)

hist = (
    alt.Chart(df_plot)
    .mark_bar(opacity=0.8)
    .encode(
        x=alt.X("residuo:Q", bin=alt.Bin(maxbins=20), title="Resíduo"),
        y=alt.Y("count():Q", title="Frequência"),
    )
    .properties(height=300)
)
d2.markdown("**Distribuição dos resíduos**")
d2.altair_chart(hist, use_container_width=True)
st.caption("Resíduos espalhados em torno de zero, sem padrão, indicam um modelo bem especificado.")

# --------------------------------------------------------------------------
# Simulador de previsão (interativo)
# --------------------------------------------------------------------------
st.subheader("🎛️ Simulador: preveja o CO₂ de um país hipotético")
st.markdown("Ajuste os valores das variáveis e veja a previsão do modelo em tempo real.")

entradas = {}
cols_sim = st.columns(min(len(variaveis_x), 3))
for i, v in enumerate(variaveis_x):
    serie = df_reg[v]
    vmin, vmax = float(serie.min()), float(serie.max())
    vmed = float(serie.median())
    col = cols_sim[i % len(cols_sim)]

    if v == "log_pib_pc":
        # Slider em dólares (mais intuitivo) e converte para log
        pib_min, pib_max = float(np.exp(vmin)), float(np.exp(vmax))
        pib_val = col.slider("PIB per capita (US$)", round(pib_min), round(pib_max),
                             round(float(np.exp(vmed))), key=f"sim_{v}")
        entradas[v] = np.log(max(pib_val, 1.0))
    elif v == "log_pib_pc_2":
        entradas[v] = entradas.get("log_pib_pc", vmed) ** 2  # derivado do PIB acima
    else:
        entradas[v] = col.slider(NOMES_VARIAVEIS.get(v, v), vmin, vmax, vmed, key=f"sim_{v}")

x_sim = np.array([1.0] + [entradas[v] for v in variaveis_x])
previsao = float(x_sim @ beta)
st.metric("CO₂ per capita previsto", f"{formatar_numero(max(previsao, 0))} t/hab")
st.caption("Estimativa do modelo para a combinação de valores escolhida acima.")

with st.expander("Ver base usada no modelo"):
    st.dataframe(df_reg, use_container_width=True)
