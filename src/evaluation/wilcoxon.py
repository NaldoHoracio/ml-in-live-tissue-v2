"""Etapa 6 — Testes de Wilcoxon.

    - intensidade: curva real média × curva prevista média (``-mean-pred``);
    - temperatura: temperaturas previstas × temperatura de referência (``-temp``).

Os resultados (``statistic`` e ``p_value``) são acrescentados aos arquivos.

Execução isolada:
    python src/evaluation/wilcoxon.py [--coef us]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from src.config import MODEL_KEYS, WILCOXON_TEMP_REF, Scenario  # noqa: E402
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.utils.file_io import read_csv, save_csv  # noqa: E402

logger = logging.getLogger(__name__)


def wilcoxon_intensity(mean_pred: pd.DataFrame) -> pd.DataFrame:
    result = stats.wilcoxon(mean_pred["yt_mean"], mean_pred["yp_mean"])
    return mean_pred.assign(statistic=result.statistic, p_value=result.pvalue)


def wilcoxon_temperature(temps: pd.DataFrame, temp_ref: float = WILCOXON_TEMP_REF) -> pd.DataFrame:
    result = stats.wilcoxon(temps["temp_result"], np.full(len(temps), temp_ref),
                            zero_method="zsplit")
    return temps.assign(statistic=result.statistic, p_value=result.pvalue)


def run_tests(scenarios: list[Scenario], model_keys=MODEL_KEYS) -> None:
    for scenario in scenarios:
        for key in model_keys:
            for kind, test in (("mean-pred", wilcoxon_intensity), ("temp", wilcoxon_temperature)):
                path = scenario.result_file(key, kind)
                df = read_csv(path).drop(columns=["statistic", "p_value"], errors="ignore")
                save_csv(test(df), path)
            logger.info("%s | %s: testes de Wilcoxon gravados", scenario.id, key)


def main(argv: list[str] | None = None) -> None:
    args = build_parser("Executa os testes de Wilcoxon.").parse_args(argv)
    setup_logging(args.verbose)
    run_tests(selected_scenarios(args), args.modelos)


if __name__ == "__main__":
    run(main)
