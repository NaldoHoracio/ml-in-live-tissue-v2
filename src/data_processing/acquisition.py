"""Etapa 1 — Aquisição: lê os dados brutos e grava entradas padronizadas.

Entrada (``data/data-raw``):
    - espectro de referência sem distorção (``I303K_compact.csv``);
    - coeficientes de absorção (μa) e espalhamento (μs) de cada tecido,
      em CSV ou nos TXT exportados.

Saída (``data/processed``):
    - ``espectro_referencia_303K.csv``;
    - ``coeficientes/<ua|us>/<cenário>.csv``.

Execução isolada:
    python src/data_processing/acquisition.py [--coef us] [--cenarios ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):   # execução direta do arquivo
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402

import pandas as pd  # noqa: E402

from src.config import (CURVE_COL, INTENSITY_COL, POINTS_PER_CURVE,  # noqa: E402
                        RAW_DIR, RAW_REFERENCE, RAW_WATER_ALPHA, REFERENCE_SPECTRUM,
                        WAVELENGTH_COL, Scenario)
from src.data_processing.spectra import curve_index  # noqa: E402
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.utils.file_io import read_any_csv, read_txt_spectra, save_csv  # noqa: E402

logger = logging.getLogger(__name__)


def load_raw_reference(path: Path = RAW_REFERENCE) -> pd.DataFrame:
    """Espectro de referência bruto, conferindo colunas e número de pontos."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)   # a pasta não vem no clone (.gitignore)
    if not path.exists():
        raise FileNotFoundError(
            f"Espectro de referência não encontrado: {path}\n"
            f"Copie os dados brutos para {RAW_DIR} (veja a seção 'Dados' do README).")
    df = read_any_csv(path)
    missing = {WAVELENGTH_COL, INTENSITY_COL} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} não tem as colunas {sorted(missing)}.")
    if len(df) != POINTS_PER_CURVE:
        raise ValueError(f"{path.name} tem {len(df)} pontos; esperado {POINTS_PER_CURVE}.")
    return df[[WAVELENGTH_COL, INTENSITY_COL]]


def water_from_alpha(path: Path = RAW_WATER_ALPHA) -> pd.DataFrame:
    """Coeficiente da água a partir do arquivo de α (cm⁻¹), convertido para mm⁻¹."""
    df = read_any_csv(path).rename(columns={"Comprimento de onda (nm)": WAVELENGTH_COL,
                                            "Alpha (cm-1)": "absortion_spectrum"})
    df["absortion_spectrum"] = df["absortion_spectrum"].mul(0.1)
    return df[[WAVELENGTH_COL, "absortion_spectrum"]]


def load_raw_coefficient(scenario: Scenario) -> tuple[pd.DataFrame, Path]:
    """Coeficiente do cenário, da primeira fonte bruta que existir."""
    kind = "absorption" if scenario.coef == "ua" else "scattering"
    for source in scenario.raw_coefficient_sources():
        if source.name == "*.txt":
            files = sorted(source.parent.glob("*.txt"))   # a ordem segue os parâmetros
            if files:
                return read_txt_spectra(files, kind), source.parent
        elif source.exists():
            if source == RAW_WATER_ALPHA:
                return water_from_alpha(source), source
            if source.suffix == ".txt":
                return read_txt_spectra([source], kind), source
            return read_any_csv(source), source
    tried = "\n  ".join(str(s) for s in scenario.raw_coefficient_sources())
    raise FileNotFoundError(f"Nenhuma fonte de coeficiente para {scenario.id}. Procurado em:\n  {tried}")


def standardize_coefficient(df: pd.DataFrame, scenario: Scenario) -> pd.DataFrame:
    """Mantém λ e o coeficiente; acrescenta ``num_curve`` (uma curva a cada 1121 pontos)."""
    col = scenario.coefficient_col
    if col not in df.columns:
        raise ValueError(f"Coluna {col!r} ausente no coeficiente de {scenario.id}.")
    out = df[[WAVELENGTH_COL, col]].reset_index(drop=True)
    if len(out) % POINTS_PER_CURVE:
        raise ValueError(f"Coeficiente de {scenario.id} tem {len(out)} linhas, "
                         f"que não formam curvas de {POINTS_PER_CURVE} pontos.")
    out[CURVE_COL] = curve_index(len(out))
    return out


def acquire(scenarios: list[Scenario]) -> None:
    """Grava a referência e os coeficientes padronizados em ``data/processed``."""
    save_csv(load_raw_reference(), REFERENCE_SPECTRUM)
    logger.info("Referência gravada em %s", REFERENCE_SPECTRUM)
    for scenario in scenarios:
        raw, source = load_raw_coefficient(scenario)
        save_csv(standardize_coefficient(raw, scenario), scenario.coefficient_file)
        logger.info("%s: coeficiente lido de %s", scenario.id, source)


def main(argv: list[str] | None = None) -> None:
    args = build_parser("Aquisição dos dados brutos.", models=False).parse_args(argv)
    setup_logging(args.verbose)
    acquire(selected_scenarios(args))


if __name__ == "__main__":
    run(main)
