# Jonatan: per-unit XGBoost + LP simulation (Kaggle v16: 0.99513, v16a: 0.99506)

- 01_eda: exploratory data analysis
- 02_modelling: LightGBM, XGBoost, per-unit models, system features, LP features (rounds 1–5), blends
- 03_neural_network: 1D CNNs (incl. the NN with system features)
- 04_simulation + lp_sim.py: weekly LP (simplified SHOP problem) -> lp_train.npz / lp_test.npz
- Short notebook 1 / 2: self-contained notebooks that reproduce v16 / v16a from the raw data

## Data (not in git)

The competition data is too large for git and is ignored by `.gitignore`. To run the notebooks, create this
layout inside `jonatan/` with the files from the course/Kaggle:

    jonatan/
      prediction_mapping.csv
      data/kernel/     Historical_day_ahead_price_2015_2025.csv, Historical_inflow_1958_2025.csv,
                       Historical_volume_2015_2024.csv, Synthetic_water_value_2015_2024.csv,
                       Unit_commitment_decisions.csv
      data/extended/   Constraint_min_flow.csv, Constraint_min_volume.csv, Tokke_Vinje_topology.yaml

Packages: `pip install -r requirements.txt` (Python 3.13; on macOS also `brew install libomp` for XGBoost).
EOF
