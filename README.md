# ml-in-live-tissue-v2

Correção espectral de nanotermômetros de LaF₃ dopados com Yb³⁺/Er³⁺ em tecidos vivos usando aprendizado de máquina.

Ao atravessar o tecido, o espectro de emissão das nanopartículas é distorcido pela absorção (μa) e pelo espalhamento (μs), conforme a lei de Beer-Lambert. Seis regressores aprendem a recuperar o espectro original a partir do espectro distorcido. Do espectro corrigido, a temperatura é estimada pela razão entre as intensidades dos picos em 1067 nm e 1340 nm (FIR).

## Estrutura

```text
ml-in-live-tissue-v2/
├── data/
│   ├── data-raw/                 # Dados brutos (não versionados)
│   └── processed/                # Entradas padronizadas e espectros gerados
├── resultados/
│   ├── tabelas/                  # CSVs: predições, métricas, σ, temperaturas
│   └── imagens/                  # Figuras
├── src/
│   ├── main.py                   # Executa o pipeline completo
│   ├── config.py                 # Caminhos, constantes, modelos e cenários
│   ├── data_processing/
│   │   ├── acquisition.py        # 1. Dados brutos → entradas padronizadas
│   │   ├── spectra.py            # 2. Alvo do treino (curva de referência repetida)
│   │   └── distortions.py        # 3. Espectros distorcidos (Beer-Lambert)
│   ├── models/
│   │   └── training.py           # 4. Treino com k-fold repetido
│   ├── evaluation/
│   │   ├── metrics.py            # 5. Métricas e σ
│   │   ├── temperature.py        # 6. Temperatura por razão de intensidades
│   │   └── wilcoxon.py           # 7. Testes de Wilcoxon
│   ├── visualization/
│   │   └── figures.py            # 8. Figuras
│   └── utils/
│       ├── cli.py                # Opções comuns de linha de comando
│       └── file_io.py            # Leitura e gravação de arquivos
├── tests/                        # Testes automatizados
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

## Instalação

Requer Python 3.10 ou superior.

```bash
git clone https://github.com/NaldoHoracio/ml-in-live-tissue-v2.git
cd ml-in-live-tissue-v2

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt    # ou requirements-dev.txt, para rodar os testes
```

## Dados

Copie os dados para `data/data-raw/`, mantendo a organização abaixo. A pasta é criada na primeira execução, se ainda não existir.

```text
data/data-raw/
├── spectrum_of_undistorted_tissues/
│   └── I303K_compact.csv                       # Espectro de referência a 303 K (1121 pontos)
├── absortion_spectrum/                         # μa
│   ├── <tecido>/<abrev>-def.csv                #   ou txt_files/<tecido>/<abrev>-def.txt
│   ├── skin/skin-var-h2o.csv                   #   ou txt_files/skin/H2O/*.txt
│   └── water/coef-water.csv                    #   ou water/alpha_interpolate_h2o_CaF2_Yb_Er-cm.csv
└── scattering_spectrum/                        # μs
    └── <tecido>/<abrev>-def.csv                #   ou txt_files/<tecido>/<abrev>-def.txt
```

Tecidos (`<tecido>` / `<abrev>`): `brain-gray-matter`/`bgm`, `breast-pre-menopause`/`bpm`, `liver`/`liver` e `skin`/`skin`.

Cada coeficiente pode estar em CSV (separador `,` ou `;`) ou nos TXT exportados (duas linhas de cabeçalho, colunas separadas por TAB e vírgula decimal). Quando os dois existem, o CSV tem preferência.

## Execução

Todos os comandos são executados a partir da raiz do projeto, com o venv ativado.

### Pipeline completo

```bash
python src/main.py
```

Opções úteis:

```bash
python src/main.py --coef us                                   # só os cenários de espalhamento
python src/main.py --cenarios us/liver ua/water                # cenários específicos
python src/main.py --modelos linR dt rf                        # modelos específicos
python src/main.py --etapas metricas temperatura wilcoxon figuras   # sem refazer o treino
python src/main.py --cenarios us/liver --folds 3 --repeticoes 2     # teste rápido
python src/main.py --help
```

### Cada etapa separadamente

Cada arquivo tem a própria função `main` e pode ser executado sozinho, com as mesmas opções `--coef`, `--cenarios` e `--modelos`:

| Etapa | Comando | Gera |
|---|---|---|
| 1. Aquisição | `python src/data_processing/acquisition.py` | `data/processed/espectro_referencia_303K.csv` e `coeficientes/` |
| 2. Alvo | `python src/data_processing/spectra.py` | `data/processed/alvo_303K_100_curvas.csv` |
| 3. Distorções | `python src/data_processing/distortions.py` | `data/processed/espectros_distorcidos/` |
| 4. Treino | `python src/models/training.py` | `resultados/tabelas/.../<modelo>-model.csv` e `-hyperparams.csv` |
| 5. Métricas | `python src/evaluation/metrics.py` | `-metrics.csv`, `-mean-pred.csv` e `-sigma.csv` |
| 6. Temperatura | `python src/evaluation/temperature.py` | `-temp.csv` |
| 7. Wilcoxon | `python src/evaluation/wilcoxon.py` | colunas `statistic` e `p_value` em `-mean-pred` e `-temp` |
| 8. Figuras | `python src/visualization/figures.py` | `resultados/imagens/` |

Opções específicas: `--folds` e `--repeticoes` (treino), `--resumo` (métricas: só mostra média ± desvio), `--salvar-calibracao` (temperatura) e `--tipos` (figuras: `distorcoes`, `intensidade`, `sigma`, `temperatura`).

Também funciona como módulo: `python -m src.main`, `python -m src.models.training` etc.

### Cenários

| ID | Tecido | Parâmetro variado |
|---|---|---|
| `ua/brain-gray-matter`, `us/brain-gray-matter` | Matéria cinzenta do cérebro | Profundidade (0–2 mm, 100 curvas) |
| `ua/breast-pre-menopause`, `us/breast-pre-menopause` | Mama pré-menopausa | Profundidade |
| `ua/liver`, `us/liver` | Fígado | Profundidade |
| `ua/skin`, `us/skin` | Pele | Profundidade |
| `ua/skin-var-h2o` | Pele | Concentração de água (0,40–0,71; L = 1 mm) |
| `ua/water` | Água | Profundidade |

### Modelos

Regressão Linear (`linR`), Árvore de Decisão (`dt`), Floresta Aleatória (`rf`), AdaBoost (`ada`), k-NN (`knn`) e MLP (`mlp`), avaliados com k-fold repetido (10 folds × 30 repetições). Os ensembles usam ⌊√(n/10)⌋ estimadores, e todos os modelos usam `random_state=42`.

## Resultados

```text
resultados/
├── tabelas/
│   └── <ua|us>/<tecido>/
│       ├── <modelo>-model.csv          # predições de todos os folds + MAE, MSE, MAPE e R² do fold
│       ├── <modelo>-hyperparams.csv    # hiperparâmetros e tempo de treino
│       ├── <modelo>-metrics.csv        # médias das métricas e σ da curva média
│       ├── <modelo>-mean-pred.csv      # curvas média real e prevista + Wilcoxon
│       ├── <modelo>-sigma.csv          # σ de cada curva prevista
│       └── <modelo>-temp.csv           # I1, I2, I3, razão, temperatura, desvio (%) e ΔT + Wilcoxon
└── imagens/
    └── <ua|us>/
        ├── distorcoes/                 # curvas distorcidas de exemplo
        ├── intensidade/                # correção espectral por modelo e grade com os seis
        ├── sigma/                      # violinos de σ por tecido e grade
        └── temperatura/                # violinos de temperatura, desvio (%) e grade de ΔT
```

No cenário `ua/skin-var-h2o`, os arquivos terminam em `-var-h2o` (tabelas) ou `-h2o` (figuras).

## Testes

```bash
pip install -r requirements-dev.txt
pytest
```

Os testes usam dados sintéticos e não dependem de `data/`.

## Autor

Edvonaldo Horácio (edvonaldohoracio@gmail.com)

## Licença

Este projeto está licenciado sob a [Licença MIT](https://opensource.org/license/mit).
