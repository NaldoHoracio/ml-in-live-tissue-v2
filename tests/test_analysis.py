import numpy as np
import pandas as pd
import pytest

from src.evaluation import metrics as evaluation, temperature, wilcoxon as statistics
from src.models import training
from src.config import (CALIBRATION_I2_I3, INTENSITY_COL,
                                   POINTS_PER_CURVE, WAVELENGTH_COL,
                                   get_scenario)
from src.data_processing.distortions import distort_varying_depth
from src.data_processing.spectra import repeat_curve


@pytest.fixture
def results(reference, coefficient) -> pd.DataFrame:
    """Predições de uma regressão linear em 3 curvas, 2 folds × 2 repetições."""
    X = distort_varying_depth(reference, coefficient, n=3)[[WAVELENGTH_COL, INTENSITY_COL]].to_numpy()
    y = repeat_curve(reference, 3)[INTENSITY_COL].to_numpy()
    model = training.build_model("linR", n_estimators=1)
    return training.repeated_kfold_predictions(X, y, model, n_splits=2, n_repeats=2)


# ---------------- treino ----------------

def test_predictions_cover_every_point_once_per_repeat(results):
    assert len(results) == 3 * POINTS_PER_CURVE * 2
    assert sorted(results["n_repeat"].unique()) == [0.0, 1.0]
    assert results.groupby("n_repeat")["k_fold"].nunique().tolist() == [2, 2]


def test_hyperparams_table_keeps_every_param():
    model = training.build_model("dt", n_estimators=1)
    table = training.hyperparams_table(model, 3725.5)
    assert len(table) == len(model.get_params()) + 1
    assert table.iloc[-1].tolist() == ["Time (h:min:s)", "1:2:5.5"]


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        training.build_model("svm", 1)


# ---------------- σ ----------------

def test_sigma_is_zero_for_identical_spectra(reference):
    i = reference[INTENSITY_COL]
    assert evaluation.sigma(i, i, reference[WAVELENGTH_COL]) == pytest.approx(0.0)


def test_sigma_ignores_scale(reference):
    i = reference[INTENSITY_COL]
    assert evaluation.sigma(i, 3 * i, reference[WAVELENGTH_COL]) == pytest.approx(0.0, abs=1e-12)


def test_sigma_by_curve_has_one_value_per_curve(results, reference):
    sigmas = evaluation.sigma_by_curve(results, reference)
    assert len(sigmas) == 3 * 2   # 3 curvas × 2 repetições
    assert (sigmas["sigma"] >= 0).all()


def test_model_metrics_names(results):
    metrics = evaluation.model_metrics(results)
    assert metrics["Metric"].tolist() == ["mae", "mse", "mape", "r2_score", "sigma"]


# ---------------- temperatura ----------------

def test_calibration_lookup_inverts_the_line():
    slope, intercept = CALIBRATION_I2_I3
    true_t = np.array([290.0, 303.0, 320.5])
    found = temperature.temperature_from_ratio(slope * true_t + intercept)
    assert np.allclose(found, true_t, atol=0.01)


def test_temperatures_by_curve_columns(results):
    temps = temperature.temperatures_by_curve(results, temp_ref=301.2)
    assert list(temps.columns) == ["num_curve", "I1", "I2", "I3", "ratioI2I3",
                                   "temp_result", "dev_per", "deltaT"]
    assert len(temps) == 6
    assert np.allclose(temps["deltaT"], np.round(np.abs(temps["temp_result"] - 301.2), 1))


# ---------------- estatística ----------------

def test_wilcoxon_temperature_adds_columns(results):
    temps = temperature.temperatures_by_curve(results, temp_ref=301.2)
    out = statistics.wilcoxon_temperature(temps)
    assert {"statistic", "p_value"} <= set(out.columns)


# ---------------- configuração ----------------

def test_result_paths_follow_naming_convention():
    s = get_scenario("ua/skin-var-h2o")
    assert s.result_file("linR", "model").name == "linR-model-var-h2o.csv"
    assert s.results_dir.parts[-3:] == ("tabelas", "ua", "skin")
    assert s.image_file("sigma", "sigma").name == "skin-sigma-h2o.png"
    assert get_scenario("us/liver").temp_ref == 304.1


def test_unknown_scenario_raises():
    with pytest.raises(ValueError):
        get_scenario("us/water")


def test_every_module_has_main():
    import importlib
    for name in ("src.main", "src.data_processing.acquisition", "src.data_processing.spectra",
                 "src.data_processing.distortions", "src.models.training",
                 "src.evaluation.metrics", "src.evaluation.temperature",
                 "src.evaluation.wilcoxon", "src.visualization.figures"):
        assert callable(importlib.import_module(name).main), name
