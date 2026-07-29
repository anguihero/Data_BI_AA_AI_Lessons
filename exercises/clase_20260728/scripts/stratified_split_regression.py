from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "taxi_trip_pricing.csv"
OUTPUT_DIR = ROOT / "data" / "splits"
TARGET_COL = "Trip_Price"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.10


def build_strata(series: pd.Series, q: int = 10) -> tuple[pd.Series, list[float]]:
    """Create quantile-based bins to stratify a continuous regression target."""
    non_null = series.dropna()
    unique_count = non_null.nunique()

    if unique_count < 2:
        raise ValueError(
            "No hay suficientes valores distintos en la variable objetivo para estratificar."
        )

    q = min(q, unique_count)
    strata, bin_edges = pd.qcut(
        non_null,
        q=q,
        labels=False,
        duplicates="drop",
        retbins=True,
    )
    return strata, bin_edges.tolist()


def summarize_strata(df: pd.DataFrame, strata_col: str, title: str) -> None:
    summary = (
        df[strata_col]
        .value_counts(normalize=True)
        .sort_index()
        .rename("proporcion")
        .mul(100)
        .round(2)
    )

    print(f"\n{title}")
    print("-" * len(title))
    print(summary.to_string())


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    total_rows = len(df)

    # Para estratificar en regresion necesitamos filas con objetivo no nulo.
    df_model = df.dropna(subset=[TARGET_COL]).copy()
    dropped_rows = total_rows - len(df_model)

    strata, bin_edges = build_strata(df_model[TARGET_COL], q=10)
    df_model["target_strata"] = strata

    dev_90, val_10 = train_test_split(
        df_model,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=df_model["target_strata"],
    )

    # El estrato solo se usa para partir; no se necesita para entrenar el modelo.
    dev_90 = dev_90.drop(columns=["target_strata"])
    val_10 = val_10.drop(columns=["target_strata"])

    dev_path = OUTPUT_DIR / "taxi_dev_90.csv"
    val_path = OUTPUT_DIR / "taxi_validation_10.csv"

    dev_90.to_csv(dev_path, index=False)
    val_10.to_csv(val_path, index=False)

    print("Split estratificado para regresion completado.")
    print(f"Filas originales: {total_rows}")
    print(f"Filas usadas para modelado (Trip_Price no nulo): {len(df_model)}")
    print(f"Filas excluidas por objetivo nulo: {dropped_rows}")
    print(f"Particion desarrollo (90%): {len(dev_90)}")
    print(f"Particion validacion (10%): {len(val_10)}")
    print(f"Archivo 90%: {dev_path}")
    print(f"Archivo 10%: {val_path}")

    summarize_strata(df_model, "target_strata", "Distribucion de estratos (dataset base)")

    dev_with_strata = dev_90.copy()
    val_with_strata = val_10.copy()
    dev_with_strata["target_strata"] = pd.cut(
        dev_with_strata[TARGET_COL],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )
    val_with_strata["target_strata"] = pd.cut(
        val_with_strata[TARGET_COL],
        bins=bin_edges,
        labels=False,
        include_lowest=True,
    )

    summarize_strata(dev_with_strata, "target_strata", "Distribucion de estratos (90%)")
    summarize_strata(val_with_strata, "target_strata", "Distribucion de estratos (10%)")


if __name__ == "__main__":
    main()
