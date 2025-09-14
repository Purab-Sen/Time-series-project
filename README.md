# 🌊 EEG Wave Next-Step Prediction & Driver Drowsiness Detection

**Summary**  
This project predicts the *next timestep* wave power for EEG bands (alpha, beta, gamma, theta) and uses those predictions / band-power sequences to detect driver drowsiness. The repo contains two Jupyter notebooks:

- `wave_prediction.ipynb` — preprocessing, band extraction, next-step forecasting (predictor).
- `decision.ipynb` — sequence generation and classification (drowsiness decision).

---

## What the notebooks do (high level)

### `wave_prediction.ipynb`
- Loads EEG signals into a DataFrame and computes sampling rate (example: `sampling_rate = math.floor(len(df)/117)` seen in the notebook).
- Designs bandpass filters (uses `scipy.signal.firwin` + `filtfilt`) and extracts standard EEG bands (delta/theta/alpha/beta/gamma).
- Implements a `bandpower` function to compute band power using windowed segments.
- Builds time-series input sequences from per-band power for **next-step prediction**.
- Trains an LSTM-based predictor (`tensorflow.keras` / `Sequential` + `LSTM`, `Dense`) and evaluates forecasting with **MSE / RMSE**.
- Saves/loads predictor and classifier models (examples in the notebook: `models/best_predictor.keras` and `./models/best_classifier.keras`).
- Provides plotting utilities to compare actual vs predicted band power.

### `decision.ipynb`
- Loads a processed dataset (`acquiredDataset.csv`) that contains band features and a label column (classification target).
- Computes per-band features (alpha, beta, theta, delta, etc.) and prepares a processed dataframe.
- Uses a `generate_sequences(dataframe, seq_length)` routine to create sliding-sequence inputs (example `SEQ_LEN = 20`).
- Defines and trains an advanced LSTM-based classifier (uses `LSTM`, `Bidirectional`, `BatchNormalization`, `Dropout`, regularizers, and `LeakyReLU`).
- Compiles classifier with `binary_crossentropy` and tracks **accuracy**; uses `classification_report` and `accuracy_score` for evaluation.
- Saves training logs (`training_log.csv`) and the best classifier model (example: `models/best_classifier.keras`).

---

## Key files & artifacts
- `wave_prediction.ipynb` — band extraction & next-step predictor (LSTM).
- `decision.ipynb` — sequence generation & drowsiness classifier (LSTM / Bidirectional).
- `acquiredDataset.csv` — processed dataset used by `decision.ipynb`.
- `models/best_predictor.keras` — saved predictor model (next-step forecasting).
- `models/best_classifier.keras` — saved classifier model (drowsiness detection).
- `training_log.csv` — example training history / logs (from classifier training).

---

## Core methods & functions (exact names used in notebooks)
- `bandpower(signal, fs, taps, window_size=...)` — computes band power per window.
- `generate_sequences(dataframe, seq_length)` — builds sliding window sequences for classification.
- `build_advanced_lstm_model(...)` — constructs the classifier (Bidirectional/LSTM + BN + Dropout + Dense).
- Models are built with `tensorflow.keras.Sequential`, `LSTM`, `Dense`, and `Bidirectional` layers.

---

## Metrics & evaluation
- **Predictor (wave next-step)**: evaluated using **MSE** and **RMSE**.
- **Classifier (drowsiness decision)**: evaluated using **accuracy** and `classification_report` (precision, recall, F1, support).

Example training hyperparameters observed in notebooks:
- Predictor: `epochs=50`, `batch_size=8` (example values found in the notebook).
- Classifier: `epochs=100`, `batch_size=64`, `SEQ_LEN=20`.

---

## How the pieces fit together (interpretation)
1. `wave_prediction.ipynb` extracts band powers (alpha, beta, gamma, theta...) and trains a next-step predictor to forecast upcoming band-power samples.
2. `decision.ipynb` creates sequence windows of band-power features and trains a classifier to output a drowsiness label (binary — e.g., alert vs drowsy).
3. The classifier works on sequences (temporal context), so combining good predictors + robust sequence features improves detection of drowsiness transitions.

---

## Notes & suggestions
- The notebooks already include useful preprocessing (FIR filters + zero-phase filtering via `filtfilt`) and bandpower extraction — good for robust EEG band isolation.
- If you want, the repo README can include:
  - A short example figure (plot of actual vs predicted alpha power).
  - A short “How to interpret the classifier output” block (e.g., probability thresholds).
  - A quick summary of final model performance (copy-paste numbers from the final evaluation cell).
- I can also create a short “Results” section that extracts the final numeric metrics (RMSE, Accuracy, Precision/Recall) from the notebooks if you want those written into the README.

---

## Author
- Purab Sen
- Ishwor Acharya
