# Jonatan: per-unit XGBoost + LP simulation (Kaggle v16: 0.99513, v16a: 0.99506)

- 01_eda: exploratory data analysis
- 02_modelling: LightGBM, XGBoost, per-unit models, system features, LP features (rounds 1–5), blends
- 03_neural_network: 1D CNNs (incl. the NN with system features)
- 04_simulation + lp_sim.py: weekly LP (simplified SHOP problem) -> lp_train.npz / lp_test.npz
- Short notebook 1 / 2: self-contained notebooks that reproduce v16 / v16a from the raw data

Put the raw data in `jonatan/data/kernel/` and `jonatan/data/extended/`, plus `prediction_mapping.csv` in `jonatan/`.
The data and generated files are not in git.
