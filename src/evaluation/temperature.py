"""Etapa 5 — Termometria por razão de intensidades (FIR, I2/I3).

A temperatura de cada curva prevista é a da razão mais próxima numa tabela
de calibração R(T) = a·T + b. Grava ``<modelo>-temp.csv`` com I1, I2, I3,
razão, temperatura, desvio percentual e ΔT.

Execução isolada:
    python src/evaluation/temperature.py [--coef us] [--salvar-calibracao]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (CALIBRATION_I2_I1, CALIBRATION_I2_I3,  # noqa: E402
                        CALIBRATION_POINTS, CALIBRATION_T_MAX,
                        CALIBRATION_T_MIN, CURVE_COL, INTENSITY_COL,
                        MODEL_KEYS, PEAK_I1, PEAK_I2, PEAK_I3, TABLES_DIR,
                        WAVELENGTH_COL, Scenario)
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.utils.file_io import read_csv, save_csv  # noqa: E402

logger = logging.getLogger(__name__)

CALIBRATIONS = {"I2/I1": CALIBRATION_I2_I1, "I2/I3": CALIBRATION_I2_I3}


def calibration_table(ratio: str = "I2/I3", t_min: float = CALIBRATION_T_MIN,
                      t_max: float = CALIBRATION_T_MAX,
                      n_points: int = CALIBRATION_POINTS) -> pd.DataFrame:
    """Tabela (temperatura, razão) da reta de calibração."""
    slope, intercept = CALIBRATIONS[ratio]
    temperature = np.linspace(t_min, t_max, n_points, endpoint=True)
    return pd.DataFrame({"temperature": temperature,
                         "ratio_emissions": slope * temperature + intercept})


def temperature_from_ratio(ratios, table: pd.DataFrame | None = None,
                           chunk_size: int = 256) -> np.ndarray:
    """Temperatura da razão mais próxima na tabela (empate: primeira linha)."""
    table = calibration_table() if table is None else table
    ref_ratios = table["ratio_emissions"].to_numpy()
    ref_temps = table["temperature"].to_numpy()
    ratios = np.atleast_1d(np.asarray(ratios, dtype=float))
    idx = np.empty(ratios.size, dtype=int)
    for start in range(0, ratios.size, chunk_size):
        block = ratios[start:start + chunk_size, None]
        idx[start:start + chunk_size] = np.abs(ref_ratios[None, :] - block).argmin(axis=1)
    return ref_temps[idx]


def peak_intensities(results: pd.DataFrame, column: str = "y_pred") -> pd.DataFrame:
    """I1, I2 e I3 de cada curva prevista (k-ésima linha de cada pico = curva k)."""
    ordered = results.sort_values(by=[WAVELENGTH_COL, "n_repeat"])
    peaks = {name: ordered.loc[ordered[WAVELENGTH_COL] == lam, column].to_numpy()
             for name, lam in (("I1", PEAK_I1), ("I2", PEAK_I2), ("I3", PEAK_I3))}
    if len({len(v) for v in peaks.values()}) != 1:
        raise ValueError("Os picos têm números de linhas diferentes.")
    return pd.DataFrame(peaks)


def temperatures_by_curve(results: pd.DataFrame, temp_ref: float,
                          table: pd.DataFrame | None = None) -> pd.DataFrame:
    df = peak_intensities(results)
    df.insert(0, "num_curve", np.arange(len(df), dtype=float))
    df["ratioI2I3"] = df["I2"] / df["I3"]
    temp = np.round(temperature_from_ratio(df["ratioI2I3"], table), 1)
    df["temp_result"] = temp
    df["dev_per"] = np.round(np.abs(temp_ref - temp) / temp_ref * 100, 1)
    df["deltaT"] = np.round(np.abs(temp - temp_ref), 1)
    return df


def temperature_of_curve(spectrum: pd.DataFrame, curve: float | None = None,
                         table: pd.DataFrame | None = None) -> float:
    """Temperatura de uma curva distorcida (padrão: a de maior distorção)."""
    curve = spectrum[CURVE_COL].max() if curve is None else curve
    sub = spectrum[spectrum[CURVE_COL] == curve]
    i2 = sub.loc[sub[WAVELENGTH_COL] == PEAK_I2, INTENSITY_COL].to_numpy()
    i3 = sub.loc[sub[WAVELENGTH_COL] == PEAK_I3, INTENSITY_COL].to_numpy()
    return float(temperature_from_ratio(i2 / i3, table)[0])


def compute_temperatures(scenarios: list[Scenario], model_keys=MODEL_KEYS) -> None:
    table = calibration_table("I2/I3")
    for scenario in scenarios:
        for key in model_keys:
            results = read_csv(scenario.result_file(key, "model"))
            save_csv(temperatures_by_curve(results, scenario.temp_ref, table),
                     scenario.result_file(key, "temp"))
            logger.info("%s | %s: temperaturas gravadas", scenario.id, key)
        dist_temp = temperature_of_curve(read_csv(scenario.distorted_spectrum), table=table)
        logger.info("%s: temperatura da curva mais distorcida, sem correção: %.1f K",
                    scenario.id, dist_temp)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Calcula a temperatura de cada curva prevista.")
    parser.add_argument("--salvar-calibracao", action="store_true",
                        help="Grava também as tabelas de calibração I2/I1 e I2/I3.")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    if args.salvar_calibracao:
        for name in CALIBRATIONS:
            path = TABLES_DIR / f"calibracao_{name.replace('/', '')}.csv"
            save_csv(calibration_table(name), path)
            logger.info("Calibração %s gravada em %s", name, path)
    compute_temperatures(selected_scenarios(args), args.modelos)


if __name__ == "__main__":
    run(main)
