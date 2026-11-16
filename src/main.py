"""Executa o pipeline completo, da aquisição dos dados às figuras.

Etapas, em ordem:
    1. aquisicao    dados brutos → entradas padronizadas      (data_processing/acquisition.py)
    2. alvo         curva de referência repetida               (data_processing/spectra.py)
    3. distorcoes   espectros distorcidos (Beer-Lambert)       (data_processing/distortions.py)
    4. treino       k-fold repetido dos seis modelos           (models/training.py)
    5. metricas     métricas, curvas médias e σ               (evaluation/metrics.py)
    6. temperatura  temperatura de cada curva prevista         (evaluation/temperature.py)
    7. wilcoxon     testes de Wilcoxon                         (evaluation/wilcoxon.py)
    8. figuras      todas as figuras                           (visualization/figures.py)

Exemplos:
    python src/main.py                                  # tudo, todos os cenários
    python src/main.py --coef us                        # só os cenários de espalhamento
    python src/main.py --etapas metricas temperatura wilcoxon figuras
    python src/main.py --cenarios us/liver --folds 3 --repeticoes 2   # teste rápido
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import logging  # noqa: E402
import time  # noqa: E402

from src.data_processing.acquisition import acquire  # noqa: E402
from src.data_processing.distortions import generate_distortions  # noqa: E402
from src.data_processing.spectra import generate_target  # noqa: E402
from src.evaluation.metrics import evaluate  # noqa: E402
from src.evaluation.temperature import compute_temperatures  # noqa: E402
from src.evaluation.wilcoxon import run_tests  # noqa: E402
from src.models.training import add_training_args, train  # noqa: E402
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.visualization.figures import generate_figures  # noqa: E402

logger = logging.getLogger("src.main")

STEPS = ("aquisicao", "alvo", "distorcoes", "treino", "metricas",
         "temperatura", "wilcoxon", "figuras")


def run_pipeline(scenarios, model_keys, steps=STEPS, n_splits: int = 10,
                 n_repeats: int = 30) -> None:
    actions = {
        "aquisicao": lambda: acquire(scenarios),
        "alvo": lambda: generate_target(),
        "distorcoes": lambda: generate_distortions(scenarios),
        "treino": lambda: train(scenarios, model_keys, n_splits, n_repeats),
        "metricas": lambda: evaluate(scenarios, model_keys),
        "temperatura": lambda: compute_temperatures(scenarios, model_keys),
        "wilcoxon": lambda: run_tests(scenarios, model_keys),
        "figuras": lambda: generate_figures(scenarios, model_keys),
    }
    total = time.perf_counter()
    for number, step in enumerate(STEPS, start=1):
        if step not in steps:
            continue
        logger.info("===== Etapa %d/%d: %s =====", number, len(STEPS), step)
        start = time.perf_counter()
        actions[step]()
        logger.info("Etapa %s concluída em %.1f s", step, time.perf_counter() - start)
    logger.info("Pipeline concluído em %.1f s", time.perf_counter() - total)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Executa o pipeline completo.")
    add_training_args(parser)
    parser.add_argument("--etapas", nargs="+", choices=STEPS, default=list(STEPS),
                        help="Etapas a executar (padrão: todas, na ordem acima).")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    run_pipeline(selected_scenarios(args), args.modelos, args.etapas,
                 args.folds, args.repeticoes)


if __name__ == "__main__":
    run(main)
