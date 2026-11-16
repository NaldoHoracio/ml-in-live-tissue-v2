"""Etapa 3 — Treinamento dos regressores com validação cruzada k-fold repetida.

Atributos: ``X = [λ, I_distorcida]``; alvo: ``y = I_sem_distorção``.
Para cada cenário e modelo grava ``<modelo>-model.csv`` (predições de todos
os folds) e ``<modelo>-hyperparams.csv`` em ``resultados/tabelas``.

Execução isolada:
    python src/models/training.py [--coef us] [--modelos linR dt] [--folds 10] [--repeticoes 30]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.base import RegressorMixin  # noqa: E402
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import (mean_absolute_error,  # noqa: E402
                             mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import RepeatedKFold  # noqa: E402
from sklearn.neighbors import KNeighborsRegressor  # noqa: E402
from sklearn.neural_network import MLPRegressor  # noqa: E402
from sklearn.tree import DecisionTreeRegressor  # noqa: E402

from src.config import (INTENSITY_COL, MODEL_KEYS, MODELS_BY_KEY,  # noqa: E402
                        N_REPEATS, N_SPLITS, RANDOM_STATE, TARGET_SPECTRUM,
                        WAVELENGTH_COL, Scenario)
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.utils.file_io import read_csv, save_csv  # noqa: E402

logger = logging.getLogger(__name__)

RESULT_COLUMNS = ["n_repeat", "k_fold", WAVELENGTH_COL, "y_pred", "y_true",
                  "mae", "mse", "mape", "r2"]


def n_estimators_for(n_samples: int) -> int:
    """Estimadores dos ensembles: ⌊√(n/10)⌋ (critério do TCC)."""
    return int(np.sqrt(n_samples / 10))


def build_model(key: str, n_estimators: int, random_state: int = RANDOM_STATE) -> RegressorMixin:
    factories = {
        "linR": lambda: LinearRegression(),
        "dt": lambda: DecisionTreeRegressor(random_state=random_state),
        "rf": lambda: RandomForestRegressor(n_estimators=n_estimators, random_state=random_state),
        "ada": lambda: AdaBoostRegressor(n_estimators=n_estimators, random_state=random_state),
        "knn": lambda: KNeighborsRegressor(),
        "mlp": lambda: MLPRegressor(random_state=random_state),
    }
    try:
        return factories[key]()
    except KeyError:
        raise ValueError(f"Modelo desconhecido: {key!r}. Opções: {list(factories)}") from None


def load_training_data(scenario: Scenario) -> tuple[np.ndarray, np.ndarray]:
    X = read_csv(scenario.distorted_spectrum)[[WAVELENGTH_COL, INTENSITY_COL]].to_numpy()
    y = read_csv(TARGET_SPECTRUM)[INTENSITY_COL].to_numpy()
    if scenario.n_training_rows is not None:
        X, y = X[:scenario.n_training_rows], y[:scenario.n_training_rows]
    if len(X) != len(y):
        raise ValueError(f"{scenario.id}: X tem {len(X)} linhas e y tem {len(y)}.")
    return X, y


def repeated_kfold_predictions(X: np.ndarray, y: np.ndarray, model: RegressorMixin,
                               n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS,
                               random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """Predições de teste de todos os folds, com as métricas de cada fold."""
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    frames = []
    for i, (train_idx, test_idx) in enumerate(rkf.split(X)):
        model.fit(X[train_idx], y[train_idx])
        y_test, y_pred = y[test_idx], model.predict(X[test_idx])
        n = len(test_idx)
        frames.append(pd.DataFrame({
            "n_repeat": np.full(n, float(i // n_splits)),
            "k_fold": np.full(n, float(i % n_splits)),
            WAVELENGTH_COL: X[test_idx, 0],
            "y_pred": y_pred.reshape(-1),
            "y_true": y_test.reshape(-1),
            "mae": mean_absolute_error(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred),
            "mape": mean_absolute_percentage_error(y_test, y_pred),
            "r2": r2_score(y_test, y_pred),
        }))
    return pd.concat(frames, ignore_index=True)[RESULT_COLUMNS]


def format_duration(seconds: float) -> str:
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes)}:{round(secs, 4)}"


def hyperparams_table(model: RegressorMixin, elapsed_seconds: float) -> pd.DataFrame:
    rows = [(str(k), str(v)) for k, v in model.get_params().items()]
    rows.append(("Time (h:min:s)", format_duration(elapsed_seconds)))
    return pd.DataFrame(rows, columns=["Hyperparam", "Value"])


def train_scenario(scenario: Scenario, model_keys=MODEL_KEYS,
                   n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS) -> None:
    X, y = load_training_data(scenario)
    n_estimators = n_estimators_for(len(y))
    logger.info("%s: %d amostras", scenario.id, len(y))
    for key in model_keys:
        model = build_model(key, n_estimators)
        start = time.perf_counter()
        predictions = repeated_kfold_predictions(X, y, model, n_splits, n_repeats)
        elapsed = time.perf_counter() - start
        save_csv(predictions, scenario.result_file(key, "model"))
        save_csv(hyperparams_table(model, elapsed), scenario.result_file(key, "hyperparams"))
        logger.info("%s | %s treinado em %s", scenario.id, MODELS_BY_KEY[key].short,
                    format_duration(elapsed))


def train(scenarios: list[Scenario], model_keys=MODEL_KEYS,
          n_splits: int = N_SPLITS, n_repeats: int = N_REPEATS) -> None:
    for scenario in scenarios:
        train_scenario(scenario, model_keys, n_splits, n_repeats)


def add_training_args(parser) -> None:
    parser.add_argument("--folds", type=int, default=N_SPLITS,
                        help=f"Número de folds (padrão: {N_SPLITS}).")
    parser.add_argument("--repeticoes", type=int, default=N_REPEATS,
                        help=f"Número de repetições (padrão: {N_REPEATS}).")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Treina os modelos com k-fold repetido.")
    add_training_args(parser)
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    train(selected_scenarios(args), args.modelos, args.folds, args.repeticoes)


if __name__ == "__main__":
    run(main)
