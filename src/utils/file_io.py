"""Leitura e gravação de arquivos (TXT exportados e CSV)."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import CSV_SEP, WAVELENGTH_COL

logger = logging.getLogger(__name__)

PathLike = str | Path

TXT_VALUE_COLUMNS = {"absorption": "absortion_spectrum", "scattering": "scattering_spectrum"}


def read_txt_spectrum(path: PathLike, kind: str = "absorption") -> pd.DataFrame:
    """Lê um TXT exportado: 2 linhas de cabeçalho, colunas separadas por TAB
    e vírgula decimal."""
    try:
        value_col = TXT_VALUE_COLUMNS[kind]
    except KeyError:
        raise ValueError(f"kind deve ser um de {list(TXT_VALUE_COLUMNS)}") from None
    df = pd.read_csv(path, sep="\t", skiprows=2, header=None,
                     names=[WAVELENGTH_COL, value_col], dtype=str)
    for col in df.columns:
        df[col] = df[col].str.strip().str.replace(",", ".").astype(np.float32)
    return df


def read_txt_spectra(paths: Iterable[PathLike], kind: str = "absorption") -> pd.DataFrame:
    """Lê vários TXT na ordem recebida (a ordem segue os parâmetros da simulação)."""
    frames = [read_txt_spectrum(p, kind) for p in paths]
    if not frames:
        raise FileNotFoundError("Nenhum arquivo TXT encontrado.")
    return pd.concat(frames, ignore_index=True)


def read_any_csv(path: PathLike) -> pd.DataFrame:
    """Lê um CSV detectando o separador (``,`` ou ``;``), para os dados brutos."""
    return pd.read_csv(path, sep=None, engine="python")


def read_csv(path: PathLike) -> pd.DataFrame:
    """Lê um CSV gerado pelo projeto."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")
    # round_trip: lê os números exatamente como foram gravados
    return pd.read_csv(path, sep=CSV_SEP, float_precision="round_trip")


def save_csv(df: pd.DataFrame, path: PathLike) -> Path:
    """Grava um CSV do projeto, criando a pasta se necessário."""
    path = Path(path)
    if df.empty:
        raise ValueError(f"DataFrame vazio; nada foi gravado em {path}.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep=CSV_SEP, index=False)
    logger.debug("Gravado: %s", path)
    return path
