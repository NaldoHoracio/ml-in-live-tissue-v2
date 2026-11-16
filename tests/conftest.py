"""Dados sintéticos no formato dos arquivos reais, para os testes."""
import numpy as np
import pandas as pd
import pytest

from src.config import INTENSITY_COL, WAVELENGTH_COL
from src.data_processing.spectra import wavelength_axis


@pytest.fixture
def wavelength() -> np.ndarray:
    return wavelength_axis()


@pytest.fixture
def reference(wavelength) -> pd.DataFrame:
    """Espectro com picos em 895, 1067 e 1340 nm."""
    lam = wavelength
    intensity = (0.02 + 0.6 * np.exp(-((lam - 895) / 12) ** 2)
                 + np.exp(-((lam - 1067) / 20) ** 2)
                 + 0.4 * np.exp(-((lam - 1340) / 25) ** 2))
    return pd.DataFrame({WAVELENGTH_COL: lam, INTENSITY_COL: intensity})


@pytest.fixture
def coefficient(wavelength) -> np.ndarray:
    return 0.3 + 0.2 * np.sin(wavelength / 40)
