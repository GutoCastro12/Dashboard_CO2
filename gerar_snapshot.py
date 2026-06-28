"""
gerar_snapshot.py
-----------------
Roda UMA vez, com internet, para baixar a base do World Bank e salvar um
snapshot em `dados/base_worldbank_snapshot.csv`.

O dashboard usa esse arquivo automaticamente como fallback caso a API esteja
fora do ar no momento da avaliação — garantindo reprodutibilidade.

Uso:
    python gerar_snapshot.py
"""

from utils_wb import _montar_base, CAMINHO_SNAPSHOT


def main():
    print("Baixando indicadores e metadados do World Bank...")
    df = _montar_base(ano_inicial=2000, ano_final=2024)
    CAMINHO_SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_SNAPSHOT, index=False, encoding="utf-8")
    print(f"Snapshot salvo em: {CAMINHO_SNAPSHOT}  ({len(df):,} linhas)")


if __name__ == "__main__":
    main()
