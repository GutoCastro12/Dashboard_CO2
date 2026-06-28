# 🌍 Desenvolvimento econômico e emissões de CO₂

Dashboard em **Streamlit** que investiga o que explica as diferenças de **CO₂ per
capita** entre países, a partir de dados abertos do **World Bank**.

## Como rodar

```bash
pip install -r requirements.txt
streamlit run Dashboard.py
```

O navegador abre o app multipágina automaticamente. As sub-páginas aparecem no menu
lateral.

## Estrutura do projeto

```
Dashboard.py            # página inicial (home)
pages/
    1_Exploracao.py     # mapa, scatter, ranking, heatmap de correlação
    2_Evolucao.py       # séries temporais, média móvel, indexação base 100
    3_Modelo.py         # regressão MQO/Ridge interativa + diagnósticos
    4_Dados.py          # base completa, filtros e download
utils_wb.py             # download, merge e funções de modelagem
gerar_snapshot.py       # (opcional) gera um CSV de fallback offline
requirements.txt
.streamlit/config.toml  # tema visual
dados/                  # snapshot opcional (gerado em runtime)
```

> Importante: para que o Streamlit reconheça o app como **multipágina**, os arquivos
> das sub-páginas ficam na pasta `pages/`. Rodar sempre a partir de `Dashboard.py`.

## Fonte e construção dos dados

Os dados vêm da **API pública do World Bank** (sem necessidade de chave). A base final
é a **união (merge/join) de várias fontes**:

| Coluna na base | Indicador | Código World Bank |
|---|---|---|
| `co2_pc` | CO₂ per capita | `EN.GHG.CO2.PC.CE.AR5` |
| `pib_pc` | PIB per capita real | `NY.GDP.PCAP.KD` |
| `urbanizacao` | População urbana (% do total) | `SP.URB.TOTL.IN.ZS` |
| `eletricidade` | Acesso à eletricidade (% da população) | `EG.ELC.ACCS.ZS` |
| `renovaveis` | Energia renovável (% do consumo final) | `EG.FEC.RNEW.ZS` |

Cada indicador é um endpoint separado, unido por `iso3` + `ano`. Em seguida, faz-se um
*join* com a base de **metadados de países** (endpoint `/country`) para acrescentar
**região** e **grupo de renda**.

### Reprodutibilidade offline

Na primeira execução com internet, o app grava um snapshot em
`dados/base_worldbank_snapshot.csv`. Se a API estiver indisponível depois, o dashboard
usa esse arquivo automaticamente. Para gerar o snapshot manualmente:

```bash
python gerar_snapshot.py
```

## Modelo

Regressão supervisionada para prever **CO₂ per capita**, implementada com `numpy`:

- **MQO (OLS)** com inferência completa: erros-padrão, estatísticas *t*, *p*-valores,
  R² e R² ajustado (cálculo via matriz `(XᵀX)⁻¹`).
- Opção de **Ridge (L2)** para reduzir variância.
- **Validação cruzada k-fold** além da divisão treino/teste.
- Comparação com um **baseline** (prever a média), para mostrar o ganho real.
- **Diagnóstico de resíduos** e **importância padronizada** das variáveis.
- Teste da **curva de Kuznets ambiental** (termo quadrático do log do PIB) com cálculo
  do ponto de virada.
- **Simulador** interativo: ajuste as variáveis e veja a previsão em tempo real.

## Elementos do dashboard (requisitos atendidos)

- Interatividade além do padrão: filtros, sliders, toggles, multiselect, simulador.
- Cinco+ elementos distintos: cards, mapa, scatter, ranking, boxplot, heatmap, séries
  temporais, tabelas e a saída do modelo.
- Saída de modelo de regressão **interativo e avaliado** (treino/teste + CV + baseline).
- App **multipágina** (`pages/`).
- Dados como **merge/join** de múltiplas fontes do World Bank.
