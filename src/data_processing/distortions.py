"""Etapa 2 — Distorções espectrais pela lei de Beer-Lambert: I = I₀·exp(−μ·L).

Para cada cenário, gera as curvas distorcidas que servem de entrada aos
modelos:
    - tecidos e água: 100 profundidades, de 0 a 2 mm;
    - pele com água variando: uma curva por concentração, com L = 1 mm.

Execução isolada:
    python src/data_processing/distortions.py [--coef ua] [--cenarios ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (CURVE_COL, DEPTH_MAX, DEPTH_MIN, INTENSITY_COL,  # noqa: E402
                        N_DEPTHS, POINTS_PER_CURVE, REFERENCE_SPECTRUM,
                        VAR_H2O_DEPTH, WAVELENGTH_COL, Scenario)
from src.data_processing.spectra import curve_index, wavelength_axis  # noqa: E402
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.utils.file_io import read_csv, save_csv  # noqa: E402

logger = logging.getLogger(__name__)


def beer_lambert(intensity, coefficient, depth) -> np.ndarray:
    """Intensidade após atravessar ``depth`` mm de um meio com coeficiente μ."""
    return np.asarray(intensity) * np.exp(-np.asarray(coefficient) * depth)


def depths(d_min: float = DEPTH_MIN, d_max: float = DEPTH_MAX, n: int = N_DEPTHS) -> np.ndarray:
    """Profundidades igualmente espaçadas.

    A soma é acumulada passo a passo, como no código original, para
    reproduzir exatamente os mesmos valores em ponto flutuante.
    """
    step = (d_max - d_min) / (n - 1)
    values, current = np.empty(n), d_min
    for i in range(n):
        values[i] = current
        current += step
    return values


def distort_varying_depth(reference: pd.DataFrame, coefficient, d_min: float = DEPTH_MIN,
                          d_max: float = DEPTH_MAX, n: int = N_DEPTHS) -> pd.DataFrame:
    """Uma curva distorcida para cada profundidade."""
    i0 = reference[INTENSITY_COL].to_numpy(dtype=float)[:POINTS_PER_CURVE]
    mu = np.asarray(coefficient, dtype=float)[:POINTS_PER_CURVE]
    curves = [beer_lambert(i0, mu, d) for d in depths(d_min, d_max, n)]
    return pd.DataFrame({WAVELENGTH_COL: np.tile(wavelength_axis(), n),
                         INTENSITY_COL: np.concatenate(curves),
                         CURVE_COL: curve_index(n * POINTS_PER_CURVE)})


def distort_varying_coefficient(reference: pd.DataFrame, coefficients,
                                depth: float) -> pd.DataFrame:
    """Uma curva distorcida para cada curva de coeficientes, com profundidade fixa."""
    mu = np.asarray(coefficients, dtype=float)
    n_rows = mu.size
    n_curves = -(-n_rows // POINTS_PER_CURVE)
    i0 = np.tile(reference[INTENSITY_COL].to_numpy(dtype=float), n_curves)[:n_rows]
    lam = np.tile(reference[WAVELENGTH_COL].to_numpy(dtype=float), n_curves)[:n_rows]
    return pd.DataFrame({WAVELENGTH_COL: lam,
                         INTENSITY_COL: beer_lambert(i0, mu, depth),
                         CURVE_COL: curve_index(n_rows)})


def distort_scenario(scenario: Scenario, reference: pd.DataFrame) -> pd.DataFrame:
    coef = read_csv(scenario.coefficient_file)[scenario.coefficient_col].to_numpy()
    if scenario.variant == "var-h2o":
        return distort_varying_coefficient(reference, coef, VAR_H2O_DEPTH)
    return distort_varying_depth(reference, coef)


def generate_distortions(scenarios: list[Scenario]) -> None:
    reference = read_csv(REFERENCE_SPECTRUM)
    for scenario in scenarios:
        save_csv(distort_scenario(scenario, reference), scenario.distorted_spectrum)
        logger.info("%s: espectros distorcidos gravados", scenario.id)


def main(argv: list[str] | None = None) -> None:
    args = build_parser("Gera os espectros distorcidos.", models=False).parse_args(argv)
    setup_logging(args.verbose)
    generate_distortions(selected_scenarios(args))


if __name__ == "__main__":
    run(main)
