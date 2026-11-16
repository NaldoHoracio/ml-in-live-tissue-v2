"""Configuração central: caminhos, constantes físicas, modelos e cenários.

Todos os caminhos partem da raiz do projeto, então os scripts funcionam
de qualquer pasta. Organização:

    data/data-raw/      dados brutos (entrada, não modificados pelo código)
    data/processed/     entradas padronizadas e espectros gerados
    resultados/tabelas/ CSVs de resultados
    resultados/imagens/ figuras
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# =============================================================
# Pastas
# =============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("MLILT_DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "data-raw"
PROCESSED_DIR = DATA_DIR / "processed"

RESULTS_DIR = Path(os.environ.get("MLILT_RESULTS_DIR", PROJECT_ROOT / "resultados"))
TABLES_DIR = RESULTS_DIR / "tabelas"
IMAGES_DIR = RESULTS_DIR / "imagens"

# =============================================================
# Dados brutos (data/data-raw), com os nomes do projeto original
# =============================================================

RAW_REFERENCE = RAW_DIR / "spectrum_of_undistorted_tissues" / "I303K_compact.csv"
RAW_ABSORPTION_DIR = RAW_DIR / "absortion_spectrum"
RAW_SCATTERING_DIR = RAW_DIR / "scattering_spectrum"
RAW_WATER_ALPHA = RAW_ABSORPTION_DIR / "water" / "alpha_interpolate_h2o_CaF2_Yb_Er-cm.csv"

# =============================================================
# Dados processados (data/processed), gerados pelo pipeline
# =============================================================

REFERENCE_SPECTRUM = PROCESSED_DIR / "espectro_referencia_303K.csv"
TARGET_SPECTRUM = PROCESSED_DIR / "alvo_303K_100_curvas.csv"
COEFFICIENTS_DIR = PROCESSED_DIR / "coeficientes"
DISTORTED_DIR = PROCESSED_DIR / "espectros_distorcidos"

CSV_SEP = ";"   # separador de todos os CSV gerados pelo projeto

# =============================================================
# Espectro
# =============================================================

TEMPERATURE = "303K"
INTENSITY_COL = f"I_{TEMPERATURE}"
WAVELENGTH_COL = "λ_(nm)"
CURVE_COL = "num_curve"

LAMBDA_START = 850.0       # nm
LAMBDA_STEP = 0.5          # nm
POINTS_PER_CURVE = 1121    # 850 a 1410 nm

PEAK_I1 = 895.0
PEAK_I2 = 1067.0
PEAK_I3 = 1340.0

# Profundidades simuladas (Beer-Lambert)
DEPTH_MIN = 0.0            # mm
DEPTH_MAX = 2.0            # mm
N_DEPTHS = 100
VAR_H2O_DEPTH = 1.0        # mm, profundidade fixa da pele com água variando
VAR_H2O_RANGE = (0.40, 0.71)

# =============================================================
# Termometria: R(T) = a·T + b (ajustes lineares feitos no OriginPro)
# =============================================================

CALIBRATION_I2_I1 = (-0.015137451, 9.725)
CALIBRATION_I2_I3 = (-0.1394, 63.63)
CALIBRATION_T_MIN = 200.0
CALIBRATION_T_MAX = 450.0
CALIBRATION_POINTS = 100_000

# Referência do Wilcoxon da temperatura: 301,2 K em todos os cenários,
# como no código original (o desvio percentual dos cenários μs usa 304,1 K).
WILCOXON_TEMP_REF = 301.2

# =============================================================
# Treinamento
# =============================================================

N_SPLITS = 10
N_REPEATS = 30
RANDOM_STATE = 42


@dataclass(frozen=True)
class ModelInfo:
    key: str        # prefixo dos arquivos
    short: str      # rótulo curto nos gráficos
    name_pt: str
    name_en: str


MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("linR", "MLR", "Regressão Linear", "Multiple Linear Regression"),
    ModelInfo("dt", "DT", "Árvore de Decisão", "Decision Tree"),
    ModelInfo("rf", "RF", "Floresta Aleatória", "Random Forest"),
    ModelInfo("ada", "Adaboost", "Adaboost", "Adaboost"),
    ModelInfo("knn", "kNN", "KNN", "k Nearest Neighbor"),
    ModelInfo("mlp", "MLP", "RNA Multi-Layer Perceptron", "Artificial Neural Network MLP"),
)
MODEL_KEYS = tuple(m.key for m in MODELS)
MODELS_BY_KEY = {m.key: m for m in MODELS}
ALPHABETICAL_MODEL_ORDER = ("ada", "dt", "knn", "linR", "mlp", "rf")

# =============================================================
# Tecidos e cenários
# =============================================================

COEF_TYPES = ("ua", "us")   # ua: absorção | us: espalhamento
COEF_COLUMNS = {"ua": "absortion_spectrum", "us": "scattering_spectrum"}


@dataclass(frozen=True)
class Tissue:
    key: str
    abbr: str
    name_pt: str
    name_en: str


TISSUES = {t.key: t for t in (
    Tissue("brain-gray-matter", "bgm", "Matéria Cinzenta do Cérebro", "Brain Gray Matter"),
    Tissue("breast-pre-menopause", "bpm", "Mama Pré-Menopausa", "Breast Pre Menopause"),
    Tissue("liver", "liver", "Fígado", "Liver"),
    Tissue("skin", "skin", "Pele", "Skin"),
    Tissue("water", "water", "Água", "Water"),
)}


@dataclass(frozen=True)
class Scenario:
    """Experimento: tecido + coeficiente (μa ou μs) + variante opcional.

    A variante ``var-h2o`` é a pele com concentração de água variando,
    com profundidade fixa.
    """

    tissue: Tissue
    coef: str
    variant: str = ""

    @property
    def id(self) -> str:
        return f"{self.coef}/{self.name}"

    @property
    def name(self) -> str:
        return f"{self.tissue.key}-{self.variant}" if self.variant else self.tissue.key

    @property
    def label_pt(self) -> str:
        return "Pele Variando Concentração de Água" if self.variant else self.tissue.name_pt

    @property
    def label_en(self) -> str:
        return "Skin with modified H2O concentration" if self.variant else self.tissue.name_en

    @property
    def varied_param(self) -> str:
        return "H2O" if self.variant else "Depth"

    @property
    def temp_ref(self) -> float:
        """Temperatura de referência (K) do desvio percentual e do ΔT."""
        return 301.2 if self.coef == "ua" else 304.1

    @property
    def coefficient_col(self) -> str:
        return COEF_COLUMNS[self.coef]

    @property
    def n_training_rows(self) -> int | None:
        """Linhas usadas no treino (``None`` = todas)."""
        return POINTS_PER_CURVE * 60 if self.variant == "var-h2o" else None

    # ----- dados brutos -----
    def raw_coefficient_sources(self) -> list[Path]:
        """Possíveis fontes brutas do coeficiente, em ordem de preferência.

        Uma entrada terminada em ``*.txt`` indica uma pasta de TXT exportados.
        """
        t = self.tissue
        if t.key == "water":
            return [RAW_ABSORPTION_DIR / "water" / "coef-water.csv", RAW_WATER_ALPHA]
        base = RAW_ABSORPTION_DIR if self.coef == "ua" else RAW_SCATTERING_DIR
        if self.variant == "var-h2o":
            return [base / "skin" / "skin-var-h2o.csv",
                    base / "txt_files" / "skin" / "H2O" / "*.txt"]
        return [base / t.key / f"{t.abbr}-def.csv",
                base / "txt_files" / t.key / f"{t.abbr}-def.txt"]

    # ----- dados processados -----
    @property
    def coefficient_file(self) -> Path:
        return COEFFICIENTS_DIR / self.coef / f"{self.name}.csv"

    @property
    def distorted_spectrum(self) -> Path:
        return DISTORTED_DIR / self.coef / f"{self.name}.csv"

    # ----- resultados -----
    @property
    def results_dir(self) -> Path:
        return TABLES_DIR / self.coef / self.tissue.key

    def result_file(self, model_key: str, kind: str) -> Path:
        """``kind``: model, hyperparams, metrics, mean-pred, sigma ou temp."""
        suffix = f"-{self.variant}" if self.variant else ""
        return self.results_dir / f"{model_key}-{kind}{suffix}.csv"

    def image_file(self, folder: str, kind: str) -> Path:
        """Ex.: ``resultados/imagens/us/sigma/liver-sigma.png``."""
        suffix = "-h2o" if self.variant else ""
        return IMAGES_DIR / self.coef / folder / f"{self.tissue.abbr}-{kind}{suffix}.png"


SCENARIOS = {s.id: s for s in (
    Scenario(TISSUES["brain-gray-matter"], "ua"),
    Scenario(TISSUES["breast-pre-menopause"], "ua"),
    Scenario(TISSUES["liver"], "ua"),
    Scenario(TISSUES["skin"], "ua"),
    Scenario(TISSUES["skin"], "ua", "var-h2o"),
    Scenario(TISSUES["water"], "ua"),
    Scenario(TISSUES["brain-gray-matter"], "us"),
    Scenario(TISSUES["breast-pre-menopause"], "us"),
    Scenario(TISSUES["liver"], "us"),
    Scenario(TISSUES["skin"], "us"),
)}


def scenarios_for(coef: str | None = None) -> list[Scenario]:
    return [s for s in SCENARIOS.values() if coef is None or s.coef == coef]


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return SCENARIOS[scenario_id]
    except KeyError:
        raise ValueError(f"Cenário desconhecido: {scenario_id!r}. "
                         f"Opções: {', '.join(SCENARIOS)}") from None
