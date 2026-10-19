# InLoQ: Hybrid Quantum Neural Network for Indoor Localisation

Trains and compares a hybrid quantum–classical neural network (InLoQ) against classical baselines (LSTM, GRU, BiLSTM) across multiple magnetic field feature configurations and multiple seeds on real-world smartphone EMFS data, with parameter efficiency analysis, quantum circuit visualisation, barren-plateau analysis, and publication-ready tables and figures.

Three orchestration scripts under `scripts/` collapse the whole workflow into three commands.

---

## Datasets, and how they're mapped

### Primary dataset (smartphone EMFS)

| Source | What it is | Registered as |
|---|---|---|
| `data/Inloq_redmi_fuldata.csv` | Main dataset, Redmi 11 smartphone, ~42,553 magnetic field samples with ground-truth `x_coord`, `y_coord` | `Inloq_redmi_fuldata` (default) |
| `data/0neplus5longwalk.csv` | OnePlus 5 long-walk recording | optional |
| `data/oneplus5-1meterhigh.csv` | OnePlus 5, 1 m height | optional |
| `data/oneplushalfmeterhigh.csv` | OnePlus 5, 0.5 m height | optional |
| `data/redme11-1meterhigh.csv` | Redmi 11, 1 m height | optional |
| `data/redme11_halfmeter.csv` | Redmi 11, 0.5 m height | optional |
| `data/redmi11longwalk.csv` | Redmi 11 long-walk recording | optional |

### Feature configurations

| Feature set name | Columns used | Model variant |
|---|---|---|
| `BT only` | `BT` | Model A (1 qubit) |
| `Bx, By, Bz` | `Bx`, `By`, `Bz` | Model B (3 qubits) |
| `Bz, Bxy, BT` | `Bz`, `Bxy`, `BT` | Model C (3 qubits) |

`Bxy` is derived as `√(Bx² + By²)` in `src/data_loader.py` if not present.

### Preprocessing

- One-Class SVM outlier removal on `BT` (ν = 0.05)
- 80 / 10 / 10 train / validation / test split, seeded by run
- `StandardScaler` normalisation (fit on train only)

---

## Architectures compared

Each feature set is trained with five model families to test whether observed quantum advantages are architecture-specific:

| Family | Architecture | Optimizer | lr |
|---|---|---|---|
| InLoQ (proposed) | `Dense(n_q) → QuantumLayer → Dense(16) → Dense(8) → Dense(2)` | Adam | 0.001 |
| Classical ANN | `Dense(w) → Dense(w/2) → Dense(2)` | Adam | 0.001 |
| LSTM | `LSTM(16) → Dropout(0.1) → Dense(8) → Dense(2)` | Adam | 0.001 |
| GRU | `GRU(16) → Dropout(0.1) → Dense(8) → Dense(2)` | Adam | 0.001 |
| BiLSTM | `BiLSTM(8+8) → Dropout(0.1) → Dense(8) → Dense(2)` | Adam | 0.001 |

The classical ANN width is auto-matched to InLoQ's parameter count in `scripts/run_experiments.py` (`find_matching_ann_width`) so the comparison is parameter-fair.

**Quantum layer details:** angle embedding (`RY`) + 2 variational layers of `Rot` gates + ring `CNOT` entanglement; `n_q = min(max(input_dim, 2), 8)`. Simulated with PennyLane `default.qubit` via `diff_method="backprop"`.

---

## Layout


InLoq/
├── requirements.txt
├── environment.yml # conda env (portable)
├── environment_full.yml # frozen conda env (exact)
├── building13.png # floor plan used in spatial plots
├── LICENSE
├── README.md
├── data/ # raw CSVs (tracked)
│ ├── Inloq_redmi_fuldata.csv # main dataset
│ └── *.csv # additional walks/heights
├── scripts/ # CLI entry points
│ ├── run_experiments.py # train all models × features × seeds
│ ├── run_experiments_with_config_file.py # same, driven by src/config.py
│ ├── reproduce_figures.py # regenerate all paper figures
│ ├── reproduce_tables.py # regenerate all LaTeX tables
│ ├── analyze_parameters.py # parameter-count comparison
│ ├── generate_circuit_diagrams.py # quantum circuit diagrams
│ └── analyze_gradient_variance.py # barren-plateau analysis
├── src/ # library code — no CLI, just importable
│ ├── config.py # paths, hyperparameters, model flags
│ ├── data_loader.py # load/clean/split + sequence prep
│ ├── models.py # InLoQ, ANN, LSTM, GRU, BiLSTM
│ ├── quantum_varied.py # variable-depth quantum layer (ablation)
│ ├── training.py # multi-run training loop
│ ├── evaluation.py # metric aggregation
│ ├── parameter_counter.py # analytical parameter counts
│ ├── plots.py # all figure-generation functions
│ ├── tables.py # all table-generation functions
│ ├── circuits_visualization.py # circuit diagram rendering
│ └── utils.py # seeds, system info, constants
├── results_InLoQ/ # created automatically (gitignored)
│ ├── Inloq/{aggregated,runs}/
│ ├── Classical_ANN/{aggregated,runs}/
│ ├── LSTM/{aggregated,runs}/
│ ├── GRU/{aggregated,runs}/
│ └── BiLSTM/{aggregated,runs}/
└── paper/ # created automatically (gitignored)
├── Plots/
│ ├── Circuit_Diagrams/
│ ├── Parameter_Analysis/
│ └── <model comparison plots>/
└── Tables/


---


---

## Setup

```bash
git clone https://github.com/Hamaad1/InLoq.git
cd InLoq

# Option 1 — conda (recommended)
conda env create -f environment.yml
conda activate inloq

# Option 2 — pip
pip install -r requirements.txt


python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# 1. Edit src/config.py to set N_RUNS, EPOCHS, and which models to train
nano src/config.py

N_RUNS     = 10      # 10 for full paper; 2 for smoke test
EPOCHS     = 100
BATCH_SIZE = 16

TRAIN_INLOQ  = True
TRAIN_ANN    = True
TRAIN_LSTM   = True
TRAIN_GRU    = True
TRAIN_BILSTM = True

#Train
python scripts/run_experiments.py #(For full training)

# Reproduce figures and tables
python scripts/reproduce_figures.py
python scripts/reproduce_tables.py

#Output folder
Outputs → paper/Plots/ and paper/Tables/.

# Parameter-count analysis
python scripts/analyze_parameters.py
Outputs → paper/Plots/Parameter_Analysis/.

# Quantum circuit diagrams
python scripts/generate_circuit_diagrams.py
Outputs → paper/Plots/Circuit_Diagrams/

# Barren-plateau / gradient variance analysis
python scripts/analyze_gradient_variance.py
Outputs → results_InLoQ/gradient_variance/


# Verify the dataset
python -c "import pandas as pd; df = pd.read_csv('data/Inloq_redmi_fuldata.csv'); print(df.shape); print(df.columns.tolist())"

Expected: ~42,553 rows, columns ['time','Bx','By','Bz','BT','x_coord','y_coord','Label','Bxy'].

# Smoke test — train a single model
python scripts/run_experiments.py --models Inloq --seeds 320 --epochs 5

Reproducibility notes
    # Every run seeds Python, NumPy, TF, and PYTHONHASHSEED via set_global_seeds().
    # Each layer's initializer receives a unique seed offset (seed, seed+1, seed+2, ...).
    # Data splits use random_state = seed, so train/valid/test is identical across models for the same seed.
    # Report mean ± std across seeds (10 runs for the paper).
    # All tables export to .tex in paper/Tables/.
    # paper/Plots/ is fully regenerable from results_InLoQ/ — no manual editing required.


Citation
If you use this code or the dataset, please cite:

@article{rafique2026inloq,
  title={InLoq: A hybrid quantum neural network for indoor localization using magnetic field signals},
  author={Rafique, Hamaad and Russo, Enrico and La Delfa, Gaetano Carmelo and Palesi, Maurizio and Torres-Sospedra, Joaqu{\'\i}n},
  journal={Computers and Electrical Engineering},
  volume={139},
  pages={111323},
  year={2026},
  publisher={Elsevier}
}

License
Released under the MIT License — see LICENSE. The dataset is provided for academic research only; contact the authors for commercial use.