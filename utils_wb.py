"""
utils_wb.py
-----------
Funções de dados e de modelagem para o dashboard de CO2 e desenvolvimento.

A base é a UNIÃO (merge/join) de várias fontes do World Bank:
  - 5 endpoints de indicadores (um por indicador), unidos por iso3 + ano;
  - 1 endpoint de metadados de países (região e grupo de renda), unido por iso3.

A parte de modelagem é implementada "na mão" com numpy (regressão por MQO),
mas com inferência estatística completa: erros-padrão, estatísticas t e
p-valores, além de validação cruzada k-fold e comparação com um modelo trivial.
"""

from __future__ import annotations

import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st

# scipy é usado apenas para p-valores exatos. Se não estiver disponível,
# caímos para uma aproximação pela normal, sem quebrar o dashboard.
try:
    from scipy import stats as _scipy_stats
    _TEM_SCIPY = True
except Exception:  # pragma: no cover
    _TEM_SCIPY = False


# ---------------------------------------------------------------------------
# Configuração de indicadores e rótulos
# ---------------------------------------------------------------------------

INDICADORES = {
    "co2_pc": {"codigo": "EN.GHG.CO2.PC.CE.AR5", "nome": "CO₂ per capita (t)"},
    "pib_pc": {"codigo": "NY.GDP.PCAP.KD", "nome": "PIB per capita real (US$)"},
    "urbanizacao": {"codigo": "SP.URB.TOTL.IN.ZS", "nome": "População urbana (% do total)"},
    "eletricidade": {"codigo": "EG.ELC.ACCS.ZS", "nome": "Acesso à eletricidade (% da população)"},
    "renovaveis": {"codigo": "EG.FEC.RNEW.ZS", "nome": "Energia renovável (% do consumo final)"},
}

NOMES_VARIAVEIS = {
    "co2_pc": "CO₂ per capita",
    "pib_pc": "PIB per capita real",
    "log_pib_pc": "Log do PIB per capita",
    "log_pib_pc_2": "Log do PIB per capita²",
    "urbanizacao": "Urbanização (%)",
    "eletricidade": "Acesso à eletricidade (%)",
    "renovaveis": "Energia renovável (%)",
}

ROTULOS_INDICADOR = {
    "co2_pc": "CO₂ per capita",
    "pib_pc": "PIB per capita",
    "urbanizacao": "Urbanização (%)",
    "eletricidade": "Acesso à eletricidade (%)",
    "renovaveis": "Energia renovável (%)",
}

# Snapshot local: gravado automaticamente quando a API responde, e usado como
# fallback quando a API estiver indisponível (garante reprodutibilidade).
CAMINHO_SNAPSHOT = Path(__file__).parent / "dados" / "base_worldbank_snapshot.csv"


# ---------------------------------------------------------------------------
# Download da API do World Bank
# ---------------------------------------------------------------------------

def _get_json(url: str, tentativas: int = 3, timeout: int = 45):
    """GET com algumas tentativas, para tolerar instabilidade da API."""
    ultimo_erro = None
    for _ in range(tentativas):
        try:
            resposta = requests.get(url, timeout=timeout)
            if resposta.status_code == 200:
                return resposta.json()
            ultimo_erro = RuntimeError(f"Status HTTP {resposta.status_code}")
        except Exception as erro:  # rede instável, timeout etc.
            ultimo_erro = erro
    raise RuntimeError(f"Falha ao acessar a API do World Bank: {ultimo_erro}")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def baixar_indicador_worldbank(
    codigo_indicador: str,
    nome_coluna: str,
    ano_inicial: int = 2000,
    ano_final: int = 2024,
) -> pd.DataFrame:
    """Baixa um indicador do World Bank para todos os países."""
    url = (
        f"https://api.worldbank.org/v2/country/all/indicator/{codigo_indicador}"
        f"?format=json&per_page=20000&date={ano_inicial}:{ano_final}"
    )
    dados = _get_json(url)
    if not isinstance(dados, list) or len(dados) < 2 or dados[1] is None:
        raise RuntimeError(f"A API não retornou dados para {nome_coluna}.")

    df = pd.DataFrame(dados[1])
    df_limpo = pd.DataFrame({
        "pais": df["country"].apply(lambda x: x["value"] if isinstance(x, dict) else np.nan),
        "iso3": df["countryiso3code"],
        "ano": pd.to_numeric(df["date"], errors="coerce"),
        nome_coluna: pd.to_numeric(df["value"], errors="coerce"),
    })
    df_limpo = df_limpo.dropna(subset=["pais", "iso3", "ano"])
    df_limpo["ano"] = df_limpo["ano"].astype(int)
    df_limpo = df_limpo.sort_values(["pais", "ano"]).reset_index(drop=True)
    return df_limpo


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def baixar_metadados_paises() -> pd.DataFrame:
    """Baixa metadados de países (região e grupo de renda)."""
    url = "https://api.worldbank.org/v2/country?format=json&per_page=400"
    dados = _get_json(url)[1]
    df = pd.DataFrame(dados)
    return pd.DataFrame({
        "iso3": df["id"],
        "pais_wb": df["name"],
        "regiao": df["region"].apply(lambda x: x["value"] if isinstance(x, dict) else np.nan),
        "grupo_renda": df["incomeLevel"].apply(lambda x: x["value"] if isinstance(x, dict) else np.nan),
    })


def _montar_base(ano_inicial: int, ano_final: int) -> pd.DataFrame:
    """Baixa indicadores + metadados e faz o merge/join completo."""
    bases = []
    for nome_coluna, info in INDICADORES.items():
        bases.append(
            baixar_indicador_worldbank(info["codigo"], nome_coluna, ano_inicial, ano_final)
        )

    df_final = bases[0]
    for base in bases[1:]:
        df_final = pd.merge(base, df_final, on=["pais", "iso3", "ano"], how="outer")

    df_paises = baixar_metadados_paises()
    df_final = pd.merge(
        df_final, df_paises[["iso3", "regiao", "grupo_renda"]], on="iso3", how="left"
    )

    # Remove agregados regionais / grupos de renda; mantém apenas países.
    df_final = df_final[df_final["regiao"] != "Aggregates"].copy()

    # Variáveis derivadas para o modelo.
    df_final["log_pib_pc"] = np.where(df_final["pib_pc"] > 0, np.log(df_final["pib_pc"]), np.nan)
    df_final["log_pib_pc_2"] = df_final["log_pib_pc"] ** 2

    return df_final.sort_values(["pais", "ano"]).reset_index(drop=True)


@st.cache_data(ttl=24 * 3600, show_spinner=True)
def carregar_base_worldbank(ano_inicial: int = 2000, ano_final: int = 2024) -> pd.DataFrame:
    """
    Carrega a base consolidada.

    Tenta a API do World Bank. Em caso de sucesso, grava um snapshot local em
    `dados/`. Se a API estiver indisponível, usa o snapshot como fallback,
    garantindo que o dashboard rode mesmo offline.
    """
    try:
        df = _montar_base(ano_inicial, ano_final)
        try:
            CAMINHO_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(CAMINHO_SNAPSHOT, index=False, encoding="utf-8")
        except Exception:
            pass  # falha ao gravar snapshot não deve quebrar o app
        return df
    except Exception as erro:
        if CAMINHO_SNAPSHOT.exists():
            st.warning(
                "A API do World Bank não respondeu agora; usando o snapshot local "
                f"em `{CAMINHO_SNAPSHOT.name}`. (Detalhe: {erro})"
            )
            return pd.read_csv(CAMINHO_SNAPSHOT)
        raise


# ---------------------------------------------------------------------------
# Filtros e preparação da base de modelagem
# ---------------------------------------------------------------------------

def filtrar_base(df: pd.DataFrame, ano: int, regioes: list[str], rendas: list[str]) -> pd.DataFrame:
    df_filtro = df[df["ano"] == ano].copy()
    if regioes:
        df_filtro = df_filtro[df_filtro["regiao"].isin(regioes)]
    if rendas:
        df_filtro = df_filtro[df_filtro["grupo_renda"].isin(rendas)]
    return df_filtro


def criar_base_modelo(
    df: pd.DataFrame,
    ano: int,
    variaveis_x: list[str],
    regioes: list[str] | None = None,
    rendas: list[str] | None = None,
) -> pd.DataFrame:
    regioes = regioes or []
    rendas = rendas or []
    colunas = ["pais", "iso3", "ano", "regiao", "grupo_renda", "co2_pc"] + variaveis_x
    df_reg = filtrar_base(df, ano, regioes, rendas)
    df_reg = df_reg[colunas].dropna(subset=["co2_pc"] + variaveis_x).copy()
    return df_reg


def separar_treino_teste(df: pd.DataFrame, prop_treino: float, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = np.arange(len(df))
    rng.shuffle(indices)
    n_treino = int(len(df) * prop_treino)
    return df.iloc[indices[:n_treino]].copy(), df.iloc[indices[n_treino:]].copy()


# ---------------------------------------------------------------------------
# Modelagem: MQO com inferência, Ridge, validação cruzada e métricas
# ---------------------------------------------------------------------------

def _norm_cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(x, dtype=float) / np.sqrt(2.0)))


def _p_valor_bicaudal(t_stats, dof: int):
    t_abs = np.abs(np.asarray(t_stats, dtype=float))
    if _TEM_SCIPY:
        return 2.0 * _scipy_stats.t.sf(t_abs, dof)
    return 2.0 * (1.0 - _norm_cdf(t_abs))


def montar_matriz(df: pd.DataFrame, variaveis_x: list[str], padronizar: bool = False):
    """Constrói a matriz de design (com intercepto). Opcionalmente padroniza X."""
    X = df[variaveis_x].to_numpy(dtype=float)
    medias = X.mean(axis=0)
    desvios = X.std(axis=0, ddof=0)
    if padronizar:
        desvios_seguro = np.where(desvios == 0, 1.0, desvios)
        X = (X - medias) / desvios_seguro
    X_design = np.column_stack([np.ones(len(X)), X])
    return X_design, medias, desvios


def ajustar_ols(X: np.ndarray, y: np.ndarray) -> dict:
    """MQO com inferência: coeficientes, erros-padrão, t, p, R² e R² ajustado."""
    n, k = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ X.T @ y
    y_pred = X @ beta
    resid = y - y_pred

    rss = float(resid @ resid)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    dof = max(n - k, 1)
    sigma2 = rss / dof

    var_beta = sigma2 * XtX_inv
    se = np.sqrt(np.maximum(np.diag(var_beta), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_stats = np.where(se > 0, beta / se, np.nan)
    p_valores = _p_valor_bicaudal(t_stats, dof)

    r2 = 1.0 - rss / ss_tot if ss_tot > 0 else np.nan
    r2_aj = 1.0 - (1.0 - r2) * (n - 1) / dof if ss_tot > 0 else np.nan

    return {
        "beta": beta, "se": se, "t": t_stats, "p": p_valores,
        "resid": resid, "y_pred": y_pred,
        "r2": r2, "r2_ajustado": r2_aj, "sigma2": sigma2,
        "n": n, "k": k, "dof": dof,
    }


def ajustar_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> np.ndarray:
    """Ridge (regularização L2). O intercepto não é penalizado."""
    k = X.shape[1]
    P = np.eye(k)
    P[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + alpha * P, X.T @ y)


def calcular_metricas(y_real: np.ndarray, y_pred: np.ndarray) -> dict:
    erro = np.asarray(y_real, float) - np.asarray(y_pred, float)
    mae = float(np.mean(np.abs(erro)))
    rmse = float(np.sqrt(np.mean(erro ** 2)))
    ss_res = float(np.sum(erro ** 2))
    ss_tot = float(np.sum((y_real - np.mean(y_real)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot != 0 else np.nan
    return {"MAE": mae, "RMSE": rmse, "R²": r2}


def metricas_baseline(y_treino: np.ndarray, y_teste: np.ndarray) -> dict:
    """Modelo trivial: prevê sempre a média do treino. Serve de referência."""
    pred = np.full_like(np.asarray(y_teste, float), float(np.mean(y_treino)))
    return calcular_metricas(y_teste, pred)


def estimar_modelo(
    df_treino: pd.DataFrame,
    df_teste: pd.DataFrame,
    variaveis_x: list[str],
    metodo: str = "ols",
    alpha: float = 0.0,
    padronizar: bool = False,
) -> dict:
    """Estima o modelo no treino e avalia no teste. Retorna tudo o necessário."""
    X_tr, medias, desvios = montar_matriz(df_treino, variaveis_x, padronizar)
    y_tr = df_treino["co2_pc"].to_numpy(float)

    # Aplica a MESMA transformação do treino ao teste (sem vazamento de dados).
    X_te_bruto = df_teste[variaveis_x].to_numpy(float)
    if padronizar:
        desvios_seguro = np.where(desvios == 0, 1.0, desvios)
        X_te_bruto = (X_te_bruto - medias) / desvios_seguro
    X_te = np.column_stack([np.ones(len(X_te_bruto)), X_te_bruto])
    y_te = df_teste["co2_pc"].to_numpy(float)

    if metodo == "ridge":
        beta = ajustar_ridge(X_tr, y_tr, alpha)
        ajuste = {"beta": beta, "se": None, "t": None, "p": None,
                  "y_pred": X_tr @ beta, "resid": y_tr - X_tr @ beta,
                  "r2": calcular_metricas(y_tr, X_tr @ beta)["R²"],
                  "r2_ajustado": np.nan, "n": len(y_tr), "k": X_tr.shape[1]}
    else:
        ajuste = ajustar_ols(X_tr, y_tr)
        beta = ajuste["beta"]

    pred_tr = X_tr @ beta
    pred_te = X_te @ beta

    return {
        "ajuste": ajuste,
        "beta": beta,
        "medias": medias,
        "desvios": desvios,
        "padronizado": padronizar,
        "metodo": metodo,
        "y_treino": y_tr, "pred_treino": pred_tr,
        "y_teste": y_te, "pred_teste": pred_te,
        "metricas_treino": calcular_metricas(y_tr, pred_tr),
        "metricas_teste": calcular_metricas(y_te, pred_te),
        "metricas_baseline": metricas_baseline(y_tr, y_te),
    }


def validacao_cruzada(
    df: pd.DataFrame,
    variaveis_x: list[str],
    k: int = 5,
    seed: int = 42,
    metodo: str = "ols",
    alpha: float = 0.0,
    padronizar: bool = False,
) -> pd.DataFrame:
    """K-fold: para cada fold treina nos demais e avalia no fold de fora."""
    n = len(df)
    k = max(2, min(k, n))
    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)
    folds = np.array_split(indices, k)

    linhas = []
    for i, fold_teste in enumerate(folds, start=1):
        mask_treino = np.ones(n, dtype=bool)
        mask_treino[fold_teste] = False
        df_tr = df.iloc[np.where(mask_treino)[0]]
        df_te = df.iloc[fold_teste]
        res = estimar_modelo(df_tr, df_te, variaveis_x, metodo, alpha, padronizar)
        m = res["metricas_teste"]
        linhas.append({"fold": i, "n_teste": len(df_te), **m})
    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Compatibilidade com a versão antiga (caso algo ainda chame estimar_mqo)
# ---------------------------------------------------------------------------

def estimar_mqo(df_treino, df_teste, variaveis_x):
    res = estimar_modelo(df_treino, df_teste, variaveis_x, metodo="ols")
    return (res["beta"], res["y_treino"], res["pred_treino"],
            res["y_teste"], res["pred_teste"])


# ---------------------------------------------------------------------------
# Formatação e exportação
# ---------------------------------------------------------------------------

def estrelas_significancia(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def formatar_numero(valor, casas: int = 2) -> str:
    if pd.isna(valor):
        return "-"
    return f"{valor:,.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def baixar_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
