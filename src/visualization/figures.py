"""Etapa 7 — Figuras.

Gera, em ``resultados/imagens/<ua|us>/``:
    distorcoes/   curvas distorcidas de exemplo
    intensidade/  correção espectral de cada modelo e grade com os seis
    sigma/        violinos de σ por tecido e grade com todos os tecidos
    temperatura/  violinos de temperatura, desvio (%) e grade de ΔT

Execução isolada:
    python src/visualization/figures.py [--coef us] [--tipos distorcoes sigma]
"""
from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import logging  # noqa: E402
from typing import Sequence  # noqa: E402

import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from src.config import (ALPHABETICAL_MODEL_ORDER, CURVE_COL, IMAGES_DIR,  # noqa: E402
                        INTENSITY_COL, MODEL_KEYS, MODELS_BY_KEY,
                        REFERENCE_SPECTRUM, VAR_H2O_RANGE, WAVELENGTH_COL,
                        Scenario)
from src.evaluation.metrics import mean_prediction  # noqa: E402
from src.utils.cli import build_parser, run, selected_scenarios, setup_logging  # noqa: E402
from src.utils.file_io import read_csv  # noqa: E402

logger = logging.getLogger(__name__)

DPI = 450
FIGURE_TYPES = ("distorcoes", "intensidade", "sigma", "temperatura")


# =============================================================
# Utilidades
# =============================================================

def _save(fig: plt.Figure, path: Path, dpi: int = DPI) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s", path)
    return path


def _bold_ticks(ax: plt.Axes) -> None:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")


def _thick_spines(ax: plt.Axes, width: float = 2) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(width)


def confidence_band(mean: pd.DataFrame, z: float = 1.96) -> tuple[pd.Series, pd.Series]:
    """Intervalo de confiança da predição média: média ± z·s/√n."""
    half = z * mean["yp_std"] / np.sqrt(mean["yp_count"])
    return mean["yp_mean"] - half, mean["yp_mean"] + half


def load_model_results(scenario: Scenario, kind: str, model_keys: Sequence[str],
                       labels: dict[str, str]) -> pd.DataFrame:
    """Concatena os resultados ``kind`` dos modelos, com uma coluna ``model``."""
    frames = []
    for key in model_keys:
        df = read_csv(scenario.result_file(key, kind))
        df["model"] = labels[key]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


# =============================================================
# Distorções
# =============================================================

def plot_distortions(scenario: Scenario) -> Path:
    """Curvas distorcidas de exemplo do cenário."""
    distorted = read_csv(scenario.distorted_spectrum)
    curves = sorted(distorted[CURVE_COL].unique())
    fig, ax = plt.subplots(figsize=(10, 7))

    if scenario.variant == "var-h2o":
        reference = read_csv(REFERENCE_SPECTRUM)
        ax.plot(reference[WAVELENGTH_COL], reference[INTENSITY_COL], color="black",
                label="Normal", linewidth=1)
        selected = [(curves[0], f"H2O: {VAR_H2O_RANGE[0]}", "blue"),
                    (curves[-1], f"H2O: {VAR_H2O_RANGE[1]}", "red")]
    else:
        selected = [(curves[0], "L = 0.0 mm", "black"),
                    (curves[len(curves) // 2 - 1], "L = 1.0 mm", "blue"),
                    (curves[-1], "L = 2.0 mm", "red")]

    for curve, label, color in selected:
        sub = distorted[distorted[CURVE_COL] == curve]
        ax.plot(sub[WAVELENGTH_COL], sub[INTENSITY_COL], color=color, label=label)
    fig.suptitle(rf"Comprimento de onda $\lambda$ vs Intensidade distorcida ({scenario.coef})")
    ax.set_title(f"Tecido: {scenario.label_pt}")
    ax.set_xlabel("λ (nm)")
    ax.set_ylabel("Intensidade")
    ax.legend()
    return _save(fig, scenario.image_file("distorcoes", "dist"))


# =============================================================
# Correção espectral
# =============================================================

def _draw_correction(ax, mean, r2, labels, linewidth, band_alpha) -> None:
    low, up = confidence_band(mean)
    x = mean[WAVELENGTH_COL]
    ax.plot(x, mean["yt_mean"], linewidth=linewidth, color="blue", label=labels[0])
    ax.plot(x, mean["yp_mean"], linewidth=linewidth, color="red", label=f"{labels[1]}{r2}")
    ax.fill_between(x, low, up, color="lightcoral", alpha=band_alpha, label=labels[2])
    ax.set_xlim(x.min(), x.max())


def plot_intensity(scenario: Scenario, model_key: str) -> Path:
    """Curva real × predição média de um modelo (em português)."""
    results = read_csv(scenario.result_file(model_key, "model"))
    mean = mean_prediction(results)
    r2 = round(results["r2"].mean(), 4)
    fig, ax = plt.subplots(figsize=(10, 10))
    _draw_correction(ax, mean, r2, ("Valor real", "Predição média\n R² médio:",
                                    "Margem do valor predito"), 1.0, 0.75)
    ax.set_ylim(results["y_true"].min(), 1.1 * results["y_true"].max())
    ax.set_title(f"Correção espectral usando {MODELS_BY_KEY[model_key].name_pt} no tecido: "
                 f"{scenario.label_pt}\nParâmetro variado: {scenario.varied_param}", fontsize=16)
    ax.set_xlabel(r"$\lambda$ (nm)", fontsize=16)
    ax.set_ylabel("Intensidade (u.a)", fontsize=16)
    ax.legend(fontsize=12)
    return _save(fig, scenario.image_file("intensidade", model_key))


def plot_intensity_grid(scenario: Scenario, model_keys: Sequence[str] = MODEL_KEYS) -> Path:
    """Correção espectral dos modelos, em grade 2×3 (em inglês)."""
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))
    fig.suptitle(f"Spectral correction {scenario.label_en}", fontsize=16, weight="semibold")
    for letter, key, ax in zip("abcdef", model_keys, axes.flat):
        results = read_csv(scenario.result_file(key, "model"))
        mean = mean_prediction(results)
        _thick_spines(ax)
        _draw_correction(ax, mean, round(results["r2"].mean(), 4),
                         ("Undistortion spectrum", "Mean prediction\nMean R²: ",
                          "Margin of predict value"), 1.25, 0.5)
        ax.set_ylim(mean["yt_mean"].min(), 1.2 * mean["yt_mean"].max())
        ax.set_title(f"({letter}) {MODELS_BY_KEY[key].name_en}", weight="bold",
                     fontsize=12, color="grey")
        ax.set_xlabel("Wavelength (nm)", fontweight="semibold", fontsize=11)
        ax.set_ylabel("Intensity (u.a.)", fontweight="semibold", fontsize=12)
        ax.legend(fontsize=8)
        _bold_ticks(ax)
    for ax in list(axes.flat)[len(model_keys):]:
        ax.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(fig, scenario.image_file("intensidade", "modelos"))


# =============================================================
# Violinos por tecido
# =============================================================

VIOLIN_SPECS = {
    "sigma": dict(kind="sigma", column="sigma", folder="sigma",
                  title="Valores de $\\sigma$ no tecido {label}", ylabel="Valor do $\\sigma$",
                  legend_title="Valor médio do $\\sigma$", unit="", decimals=4,
                  max_value=2.0, density_norm="width", fontsize=14, xlabel="", english=False),
    "temp": dict(kind="temp", column="temp_result", folder="temperatura",
                 title="Valores de temperatura no tecido {label}", ylabel="Temperatura (em K)",
                 legend_title="Temperatura média (em K)", unit=" K", decimals=1,
                 max_value=None, density_norm="width", fontsize=16, xlabel="", english=False),
    "dev-per": dict(kind="temp", column="dev_per", folder="temperatura",
                    title="Percentage variation in temperature {label}",
                    ylabel="Variation temperature (in %)",
                    legend_title="Mean variation in temperature (in %)", unit=" %",
                    decimals=2, max_value=None, density_norm="count", fontsize=14,
                    xlabel="Algorithm", english=True),
}


def _legend_values(df: pd.DataFrame, spec: dict, temp_ref: float) -> pd.Series:
    if spec["column"] == "dev_per":
        mean_temp = df.groupby("model", sort=False)["temp_result"].mean().round(1)
        return (np.abs(mean_temp - temp_ref) / temp_ref * 100).round(spec["decimals"])
    return df.groupby("model", sort=False)[spec["column"]].mean().round(spec["decimals"])


def plot_violin(scenario: Scenario, what: str,
                model_keys: Sequence[str] = ALPHABETICAL_MODEL_ORDER) -> Path:
    """Violino de σ (``sigma``), temperatura (``temp``) ou desvio (``dev-per``)."""
    spec = VIOLIN_SPECS[what]
    labels = {k: MODELS_BY_KEY[k].short for k in model_keys}
    if spec["english"] and "mlp" in labels:
        labels["mlp"] = "ANN MLP"
    df = load_model_results(scenario, spec["kind"], model_keys, labels)
    order = [labels[k] for k in model_keys]
    legend_values = _legend_values(df, spec, scenario.temp_ref)
    data = df if spec["max_value"] is None else df[df[spec["column"]] <= spec["max_value"]]
    label = scenario.label_en if spec["english"] else scenario.label_pt

    with sns.axes_style("white"), sns.plotting_context(font_scale=0.8):
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.violinplot(data=data, x="model", y=spec["column"], order=order, hue="model",
                       hue_order=order, legend=False, density_norm=spec["density_norm"],
                       bw_method="silverman", orient="v", linewidth=0.5, ax=ax)
        ax.set_title(spec["title"].format(label=label), fontsize=spec["fontsize"])
        ax.set_xlabel(spec["xlabel"], fontsize=spec["fontsize"])
        ax.set_ylabel(spec["ylabel"], fontsize=spec["fontsize"])
        colors = sns.color_palette()
        handles = [mpatches.Patch(color=colors[i],
                                  label=f"{name}: {legend_values[name]}{spec['unit']}")
                   for i, name in enumerate(order)]
        ax.legend(handles=handles, loc="upper right", shadow=True,
                  title=spec["legend_title"], fontsize=10)
        _bold_ticks(ax)
    return _save(fig, scenario.image_file(spec["folder"], what))


# =============================================================
# Grades com todos os tecidos de um coeficiente
# =============================================================

GRID_SPECS = {
    "sigma": dict(kind="sigma", column="sigma", max_value=0.6, ylabel=r"$\sigma$",
                  title=r"Integral absolute difference ($\sigma$) with the $\mu${letter}",
                  folder="sigma", name="grade-sigma.png"),
    "deltaT": dict(kind="temp", column="deltaT", max_value=None, ylabel=r"$\Delta$T",
                   title=r"Absolute temperature difference (K) with the $\mu${letter}",
                   folder="temperatura", name="grade-delta-temp.png"),
}


def plot_violin_grid(scenarios: Sequence[Scenario], what: str,
                     model_keys: Sequence[str] = MODEL_KEYS) -> Path:
    """Um painel por tecido, com os modelos lado a lado."""
    spec = GRID_SPECS[what]
    coef = scenarios[0].coef
    n_cols = 3 if len(scenarios) > 4 else 2
    n_rows = int(np.ceil(len(scenarios) / n_cols))
    labels = {k: ("ANN MLP" if k == "mlp" else MODELS_BY_KEY[k].short) for k in model_keys}
    order = [labels[k] for k in model_keys]

    with sns.axes_style("white"), sns.plotting_context(font_scale=0.85):
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows),
                                 squeeze=False)
        fig.suptitle(spec["title"].format(letter=coef[1]), fontsize=18, weight="semibold")
        for letter, scenario, ax in zip("abcdefghij", scenarios, axes.flat):
            df = load_model_results(scenario, spec["kind"], model_keys, labels)
            if spec["max_value"] is not None:
                df = df[df[spec["column"]] <= spec["max_value"]]
            _thick_spines(ax)
            sns.violinplot(data=df, x="model", y=spec["column"], order=order, hue="model",
                           hue_order=order, legend=False, density_norm="width",
                           bw_method="silverman", orient="v", linewidth=0.5, ax=ax)
            ax.set_title(f"({letter}) {scenario.label_en}", fontsize=12, weight="semibold")
            ax.set_xlabel("")
            ax.set_ylabel(spec["ylabel"], fontsize=14, weight="bold")
            _bold_ticks(ax)
        for ax in list(axes.flat)[len(scenarios):]:
            ax.set_visible(False)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _save(fig, IMAGES_DIR / coef / spec["folder"] / spec["name"])


# =============================================================
# Execução
# =============================================================

def generate_figures(scenarios: list[Scenario], model_keys: Sequence[str] = MODEL_KEYS,
                     types: Sequence[str] = FIGURE_TYPES) -> None:
    plt.switch_backend("Agg")   # grava arquivos sem abrir janelas
    alphabetical = [k for k in ALPHABETICAL_MODEL_ORDER if k in model_keys]
    for scenario in scenarios:
        if "distorcoes" in types:
            plot_distortions(scenario)
        if "intensidade" in types:
            for key in model_keys:
                plot_intensity(scenario, key)
            plot_intensity_grid(scenario, model_keys)
        if "sigma" in types:
            plot_violin(scenario, "sigma", alphabetical)
        if "temperatura" in types:
            plot_violin(scenario, "temp", alphabetical)
            plot_violin(scenario, "dev-per", alphabetical)

    for coef in sorted({s.coef for s in scenarios}):
        group = [s for s in scenarios if s.coef == coef]
        if "sigma" in types:
            plot_violin_grid(group, "sigma", model_keys)
        if "temperatura" in types:
            plot_violin_grid(group, "deltaT", model_keys)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser("Gera as figuras.")
    parser.add_argument("--tipos", nargs="+", choices=FIGURE_TYPES, default=list(FIGURE_TYPES),
                        help="Tipos de figura (padrão: todos).")
    args = parser.parse_args(argv)
    setup_logging(args.verbose)
    generate_figures(selected_scenarios(args), args.modelos, args.tipos)


if __name__ == "__main__":
    run(main)
