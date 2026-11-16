"""Etapa 4 — Métricas: médias do sklearn e σ espectral.

σ = ∫ |I_ref/A_ref − I_pred/A_pred| dλ, em que A é a área sob cada curva.

Para cada cenário e modelo grava, em ``resultados/tabelas``:
    ``-metrics`` (médias + σ da curva média), ``-mean-pred`` (curvas médias)
    e ``-sigma`` (σ de cada curva prevista).

Execução isolada:
    python src/evaluation/metrics.py [--coef us] [--resumo]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (INTENSITY_COL, MODEL_KEYS, POINTS_PER_CURVE,  # noqa: E402
                        REFERENCE_SPECTRUM, WAVELENGTH_COL, Scenario)
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.utils.file_io import read_csv, save_csv  # noqa: E402

logger = logging.getLogger(__name__)

trapezoid = getattr(np, "trapezoid", None) or np.trapz   # NumPy 2 renomeou trapz
METRIC_COLUMNS = ("mae", "mse", "mape", "r2")


# ----------------------------- σ -----------------------------

def normalize_by_area(intensity, wavelength) -> np.ndarray:
    """Divide pela área sob a curva (por linha, se 2D)."""
    intensity = np.asarray(intensity, dtype=float)
    area = trapezoid(intensity, x=np.asarray(wavelength, dtype=float), axis=-1)
    return intensity / np.expand_dims(area, -1)


def absolute_normalized_difference(reference, predicted, wavelength) -> np.ndarray:
    return np.abs(normalize_by_area(reference, wavelength)
                  - normalize_by_area(predicted, wavelength))


def sigma(reference, predicted, wavelength):
    """σ entre a referência e uma ou várias (linhas) curvas previstas."""
    diff = absolute_normalized_difference(reference, predicted, wavelength)
    return trapezoid(diff, x=np.asarray(wavelength, dtype=float), axis=-1)


# ----------------------- agregações --------------------------

def mean_prediction(results: pd.DataFrame) -> pd.DataFrame:
    """Média, desvio e contagem de y_true e y_pred por λ."""
    grouped = results.groupby(WAVELENGTH_COL)[["y_true", "y_pred"]].agg(["mean", "std", "count"])
    grouped.columns = [f"{'yt' if c == 'y_true' else 'yp'}_{stat}" for c, stat in grouped.columns]
    return grouped.reset_index()


def mean_prediction_table(results: pd.DataFrame) -> pd.DataFrame:
    return mean_prediction(results)[[WAVELENGTH_COL, "yp_mean", "yt_mean"]].copy()


def model_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Média das métricas de todos os folds + σ entre as curvas médias."""
    mean = mean_prediction(results)
    values = {m: round(results[m].mean(), 8) for m in METRIC_COLUMNS}
    values["sigma"] = round(float(sigma(mean["yt_mean"], mean["yp_mean"], mean[WAVELENGTH_COL])), 8)
    return pd.DataFrame({"Metric": [{"r2": "r2_score"}.get(k, k) for k in values],
                         "Value": list(values.values())})


def predictions_by_curve(results: pd.DataFrame, n_points: int = POINTS_PER_CURVE) -> np.ndarray:
    """Predições em matriz (n_curvas, n_pontos).

    Ordena por (λ, n_repeat); a k-ésima linha de cada λ forma a curva k.
    """
    y_pred = results.sort_values(by=[WAVELENGTH_COL, "n_repeat"])["y_pred"].to_numpy()
    if y_pred.size % n_points:
        raise ValueError("O número de predições não é múltiplo do número de pontos por curva.")
    return y_pred.reshape(n_points, -1).T


def sigma_by_curve(results: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    curves = predictions_by_curve(results)
    values = sigma(reference[INTENSITY_COL].to_numpy(dtype=float), curves,
                   reference[WAVELENGTH_COL].to_numpy(dtype=float))
    return pd.DataFrame({"num_curve": np.arange(len(curves), dtype=float), "sigma": values})


def metrics_summary(results: pd.DataFrame, sigmas: pd.DataFrame,
                    sigma_max: float = 5.0) -> dict[str, tuple[float, float]]:
    """Média e desvio das métricas; σ acima de ``sigma_max`` é descartado (como no original)."""
    summary = {m.upper(): (results[m].mean(), results[m].std()) for m in METRIC_COLUMNS}
    kept = sigmas.loc[sigmas["sigma"] <= sigma_max, "sigma"]
    summary["SIGMA"] = (kept.mean(), kept.std())
    return summary


# ------------------------- execução --------------------------

def evaluate(scenarios: list[Scenario], model_keys=MODEL_KEYS) -> None:
    reference = read_csv(REFERENCE_SPECTRUM)
    for scenario in scenarios:
        for key in model_keys:
            results = read_csv(scenario.result_file(key, "model"))
            save_csv(model_metrics(results), scenario.result_file(key, "metrics"))
            save_csv(mean_prediction_table(results), scenario.result_file(key, "mean-pred"))
            save_csv(sigma_by_curve(results, reference), scenario.result_file(key, "sigma"))
            logger.info("%s | %s: métricas e σ gravados", scenario.id, key)


def print_summary(scenarios: list[Scenario], model_keys=MODEL_KEYS) -> None:
    for scenario in scenarios:
        for key in model_keys:
            summary = metrics_summary(read_csv(scenario.result_file(key, "model")),
                                      read_csv(scenario.result_file(key, "sigma")))
            print(f"\n{scenario.id} | {key}")
            for name, (mean, std) in summary.items():
                print(f"  {name:6s} {mean: .4f} ± {std: .4f}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Calcula métricas, curvas médias e σ.")
    parser.add_argument("--resumo", action="store_true",
                        help="Só mostra média ± desvio das métricas já calculadas.")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    scenarios = selected_scenarios(args)
    if not args.resumo:
        evaluate(scenarios, args.modelos)
    print_summary(scenarios, args.modelos)


if __name__ == "__main__":
    run(main)
