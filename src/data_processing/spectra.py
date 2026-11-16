"""Curvas espectrais: eixo λ, repetição de curvas e o alvo do treino.

O alvo é a curva de referência sem distorção repetida N vezes, uma para
cada curva distorcida.

Execução isolada (gera o alvo):
    python src/data_processing/spectra.py [--curvas 100]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import (INTENSITY_COL, LAMBDA_START, LAMBDA_STEP,  # noqa: E402
                        N_DEPTHS, POINTS_PER_CURVE, REFERENCE_SPECTRUM,
                        TARGET_SPECTRUM, WAVELENGTH_COL)
from src.utils.cli import build_parser, run, setup_logging  # noqa: E402
from src.utils.file_io import read_csv, save_csv  # noqa: E402

logger = logging.getLogger(__name__)


def wavelength_axis(n_points: int = POINTS_PER_CURVE, start: float = LAMBDA_START,
                    step: float = LAMBDA_STEP, dtype=np.float64) -> np.ndarray:
    """λ de uma curva: start, start+step, ..."""
    return (start + step * np.arange(n_points)).astype(dtype)


def repeated_wavelength_axis(n_curves: int, n_points: int = POINTS_PER_CURVE) -> pd.DataFrame:
    """Eixo λ repetido ``n_curves`` vezes."""
    return pd.DataFrame({WAVELENGTH_COL: np.tile(wavelength_axis(n_points, dtype=np.float32),
                                                 n_curves)})


def curve_index(n_rows: int, n_points: int = POINTS_PER_CURVE) -> np.ndarray:
    """Número da curva de cada linha (0 nas primeiras ``n_points``, 1 nas seguintes...)."""
    return (np.arange(n_rows) // n_points).astype(float)


def curve_matrix(values, n_points: int = POINTS_PER_CURVE) -> np.ndarray:
    """Vetor de curvas empilhadas → matriz (n_curvas, n_pontos)."""
    values = np.asarray(values)
    if values.size % n_points:
        raise ValueError(f"{values.size} valores não formam curvas de {n_points} pontos.")
    return values.reshape(-1, n_points)


def repeat_curve(reference: pd.DataFrame, n_curves: int,
                 intensity_col: str = INTENSITY_COL) -> pd.DataFrame:
    """Empilha ``n_curves`` cópias de uma curva."""
    return pd.DataFrame({
        WAVELENGTH_COL: np.tile(reference[WAVELENGTH_COL].to_numpy(dtype=float), n_curves),
        intensity_col: np.tile(reference[intensity_col].to_numpy(dtype=float), n_curves),
    })


def generate_target(n_curves: int = N_DEPTHS) -> Path:
    """Grava o alvo do treino a partir da referência processada."""
    target = repeat_curve(read_csv(REFERENCE_SPECTRUM), n_curves)
    save_csv(target, TARGET_SPECTRUM)
    logger.info("Alvo com %d curvas gravado em %s", n_curves, TARGET_SPECTRUM)
    return TARGET_SPECTRUM


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Gera o alvo do treino (curva de referência repetida).",
                          scenarios=False, models=False)
    parser.add_argument("--curvas", type=int, default=N_DEPTHS,
                        help=f"Número de cópias da curva (padrão: {N_DEPTHS}).")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    generate_target(args.curvas)


if __name__ == "__main__":
    run(main)
