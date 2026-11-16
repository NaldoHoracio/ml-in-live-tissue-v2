"""Utilidades comuns às funções ``main`` dos scripts."""
from __future__ import annotations

import argparse
import logging
import sys
from typing import Callable

from src.config import COEF_TYPES, MODEL_KEYS, Scenario, get_scenario, scenarios_for

logger = logging.getLogger("src")


def build_parser(description: str, scenarios: bool = True,
                 models: bool = True) -> argparse.ArgumentParser:
    """Parser com as opções comuns: ``--coef``, ``--cenarios``, ``--modelos`` e ``-v``."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Mostra mensagens de depuração.")
    if scenarios:
        parser.add_argument("--coef", choices=COEF_TYPES,
                            help="Só os cenários de absorção (ua) ou de espalhamento (us).")
        parser.add_argument("--cenarios", nargs="+", metavar="ID",
                            help="Cenários específicos, ex.: us/liver ua/skin-var-h2o.")
    if models:
        parser.add_argument("--modelos", nargs="+", choices=MODEL_KEYS,
                            default=list(MODEL_KEYS), help="Modelos (padrão: todos).")
    return parser


def selected_scenarios(args: argparse.Namespace) -> list[Scenario]:
    if getattr(args, "cenarios", None):
        return [get_scenario(s) for s in args.cenarios]
    return scenarios_for(getattr(args, "coef", None))


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s",
                        datefmt="%H:%M:%S", force=True)


def run(main: Callable[[], None]) -> None:
    """Executa ``main`` e converte erros esperados em mensagem + código de saída 1."""
    try:
        main()
    except (FileNotFoundError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(1)
