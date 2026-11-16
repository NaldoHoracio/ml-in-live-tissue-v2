import numpy as np
import pytest

from src.config import (CURVE_COL, INTENSITY_COL, POINTS_PER_CURVE,
                                   WAVELENGTH_COL)
from src.data_processing.distortions import (depths, distort_varying_coefficient,
                                        distort_varying_depth)
from src.data_processing.spectra import (curve_index, curve_matrix,
                                    repeated_wavelength_axis, repeat_curve,
                                    wavelength_axis)


def test_wavelength_axis_covers_850_to_1410():
    axis = wavelength_axis()
    assert len(axis) == POINTS_PER_CURVE
    assert axis[0] == 850.0 and axis[-1] == 1410.0


def test_repeated_axis_restarts_each_curve():
    axis = repeated_wavelength_axis(3)[WAVELENGTH_COL].to_numpy()
    assert len(axis) == 3 * POINTS_PER_CURVE
    assert axis[POINTS_PER_CURVE] == 850.0


def test_repeat_curve_stacks_identical_copies(reference):
    stacked = repeat_curve(reference, 4)
    matrix = curve_matrix(stacked[INTENSITY_COL].to_numpy())
    assert matrix.shape == (4, POINTS_PER_CURVE)
    assert np.all(matrix == matrix[0])


def test_curve_index():
    idx = curve_index(2 * POINTS_PER_CURVE + 1)
    assert idx[0] == 0 and idx[POINTS_PER_CURVE] == 1 and idx[-1] == 2


def test_depths_are_evenly_spaced():
    values = depths(0.0, 2.0, 100)
    assert values[0] == 0.0
    assert values[-1] == pytest.approx(2.0)
    assert np.allclose(np.diff(values), 2.0 / 99)


def test_zero_depth_keeps_spectrum(reference, coefficient):
    distorted = distort_varying_depth(reference, coefficient, n=5)
    first = distorted[distorted[CURVE_COL] == 0][INTENSITY_COL].to_numpy()
    assert np.allclose(first, reference[INTENSITY_COL])


def test_beer_lambert_attenuation(reference, coefficient):
    distorted = distort_varying_depth(reference, coefficient, d_max=2.0, n=3)
    last = distorted[distorted[CURVE_COL] == 2][INTENSITY_COL].to_numpy()
    expected = reference[INTENSITY_COL].to_numpy() * np.exp(-coefficient * 2.0)
    assert np.allclose(last, expected)


def test_varying_coefficient_one_curve_per_coefficient(reference, coefficient):
    coefs = np.concatenate([coefficient, 2 * coefficient])
    distorted = distort_varying_coefficient(reference, coefs, depth=1.0)
    assert distorted[CURVE_COL].nunique() == 2
    second = distorted[distorted[CURVE_COL] == 1][INTENSITY_COL].to_numpy()
    assert np.allclose(second, reference[INTENSITY_COL] * np.exp(-2 * coefficient))


def test_read_txt_spectrum_parses_decimal_comma(tmp_path):
    from src.utils.file_io import read_txt_spectrum
    path = tmp_path / "espectro.txt"
    path.write_text("cabeçalho 1\ncabeçalho 2\n850,0\t0,125\n850,5\t0,250\n")
    df = read_txt_spectrum(path, "scattering")
    assert list(df.columns) == [WAVELENGTH_COL, "scattering_spectrum"]
    assert df.iloc[1].tolist() == [850.5, 0.25]
