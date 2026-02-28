# Interpretable GCN for CVD Risk Prediction in T2DM

This repository contains a PyTorch Geometric implementation of the framework described in **"Interpretable Graph Convolutional Networks for cardiovascular disease risk prediction in patients with Type 2 Diabetes Mellitus"**.

The model utilizes a **Graph Convolutional Network (GCN)** formulated as an **anomaly detection task** to handle severe class imbalance. It constructs a population graph where patients are nodes, learns a "normal" embedding space for non-CVD patients, and identifies high-risk patients as deviations from this center.

##Repository Structure

```text
gnn-cvd-risk/
├── data/
│   └── dataset.csv          # [ Create this folder and place your private data here]
├── src/
│   ├── dataset.py              # Graph construction & Distance metric 
│   ├── model.py                # Two-layer GCN 
│   ├── loss.py                 # Hypersphere + Differential AUC Loss 
│   ├── utils.py                # Probability calibration  & Metrics
│   └── interpretability.py     # RuleFit Surrogate Model wrapper
|   └── generate_dummy_data.py  # Dummy data generation
├── main.py                  # Entry point: Data loading, Training, Evaluation
├── requirements.txt         # Dependencies
└── README.md                # This file
```

##  Installation

1. **Clone the repository:**
   ```bash
   git clone <repo_url>
   cd gnn-cvd-risk
   ```

2. **Install dependencies:**
   It is recommended to use a virtual environment (Conda or venv).
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have a version of `torch` and `torch_geometric` compatible with your CUDA version if using GPU.*

## Data Preparation

Since the dataset is private, this repository is designed to be **data-agnostic**. You must provide your own dataset in CSV format.

1. **File Location:** Place your `.csv` file in the `data/` directory (e.g., `data/my_patients.csv`).
2. **Format:**
   - Rows: Individual patients.
   - Columns: Clinical features (continuous and categorical) + 1 Target column.
   - **Target Column:** Must be binary integers:
     - `0`: Normal / Control (Non-CVD)
     - `1`: Anomaly / Case (CVD)
   - **Missing Values:** The current implementation assumes no missing values (impute them before running).

## Testing with Dummy Data

If you do not have access to a private medical dataset but want to test the methodology, we provide a script to generate a synthetic dataset. This synthetic data statistically mirrors the population described in the original paper (N=560, ~7% CVD prevalence, matching feature distributions).

1. **Generate the data:**
   ```bash
   python generate_dummy_data.py

### Feature Auto-Detection
The script `main.py` currently uses a heuristic to distinguish categorical from continuous variables:
- **Categorical:** Columns with < 10 unique values.
- **Continuous:** All other feature columns.

*If your data requires specific column definitions, edit the `cat_cols` and `cont_cols` lists in `main.py` manually.*

##  Usage

To train the model and evaluate it using Stratified Cross-Validation:

```bash
python main.py --data_path data/dataset.csv --target_col CVD_Outcome
```

### Command Line Arguments

You can tune the hyperparameters to match the settings described in the paper or to suit your specific dataset:

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--data_path` | `data/dataset.csv` | Path to your input CSV file. |
| `--target_col` | `CVD_Outcome` | Name of the target variable column. |
| `--gamma` | `0.6` | Threshold for graph adjacency pruning. |
| `--lambda_loss` | `1.0` | Weight $\lambda$ for the Differential AUC loss term. |
| `--hidden_dim` | `16` | Number of neurons in the hidden GCN layer. |
| `--output_dim` | `8` | Dimension of the output embedding space. |
| `--dropout` | `0.5` | Dropout rate. |
| `--lr` | `0.01` | Learning rate for Adam optimizer. |
| `--epochs` | `200` | Number of training epochs per fold. |

### Example with Custom Hyperparameters

```bash
python main.py \
  --data_path data/my_data.csv \
  --target_col outcome \
  --gamma 0.5 \
  --hidden_dim 32 \
  --lambda_loss 10.0 \
  --epochs 500
```

###  Interpretability
The repository includes a wrapper for **RuleFit** (in `src/interpretability.py`). This fits a surrogate model (Decision Rules + Linear Model) on the GCN's predictions to provide human-readable explanations (e.g., *IF Glucose > 160 AND Age > 60 THEN Risk High*).
