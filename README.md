# Brugada Syndrome Detection via Deep Learning on ECG Signals

Detection of **Brugada Syndrome** from 12-lead ECG recordings using a dual-input CNN + LSTM architecture trained on the [Brugada-HUCA dataset](https://physionet.org/content/brugada-ecg-huca/1.0.0/) from PhysioNet.

---

## Results

Best model: **CNN Dual Input v3.3** (LSTM + statistical features branch)

| Model | AUC-ROC | Recall | Precision | FP | FN |
|-------|---------|--------|-----------|----|----|
| CNN Base (v1) | — | 0.933 | 0.424 | 19 | 1 |
| CNN Residual + LSTM (v2) | 0.953 | 0.867 | 0.565 | 11 | 2 |
| CNN Dual Input (v3.1) | 0.944 | 0.867 | 0.765 | 4 | 2 |
| **CNN Dual + fine-tune (v3.3)** | **0.912** | **0.933** | **0.737** | **5** | **1** |

Test set: 73 patients (58 controls, 15 Brugada) — never seen during training.

Clinical priority is **Recall** (sensitivity): missing a Brugada patient is the most dangerous error. The v3.3 model achieves Recall=0.933 with only 1 false negative.

---

## Dataset

**Brugada-HUCA** (PhysioNet): 363 patients from Hospital Universitario Central de Asturias.
- 76 Brugada positive, 287 controls
- 12-lead ECG, 100 Hz sampling rate, 12-second recordings
- Only leads V1, V2, V3 are used (clinically relevant for Brugada pattern)

Download:
```bash
wget -r -N -c -np https://physionet.org/files/brugada-ecg-huca/1.0.0/ -P brugada_data/
```

---

## Project Structure

```
brugada_syndrom/
│
├── FASE1/
│   ├── fase1_generar_dataset.py      # Load raw .dat/.hea files, extract V1/V2/V3
│   └── fase1_preprocesamiento.py     # Notch filter (50Hz) + bandpass (0.5–40Hz)
│
├── FASE2/
│   ├── fase2_balanceo_v3.py          # Stratified split + 8-technique augmentation
│   │                                  # (MixUp, CutMix, noise, shift, scaling...)
│   ├── fase3_modelo_v3.py            # CNN Dual Input v3.1 — main training script
│   └── fase3_modelo_v33_finetune.py  # v3.3 fine-tuning from v3.1 checkpoint
│
├── results/
│   ├── resultados_brugada_v31.png    # Training curves + ROC + confusion matrix
│   ├── resultados_brugada_v33.png    # Fine-tuning results
│   └── verificacion_augmentation.png # Visual check of augmented signals
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Architecture

The model uses **dual inputs** processed in parallel:

```
Signal (1200, 3)              Statistical features (18,)
       │                               │
CNN Residual Blocks            MLP (Dense 32 → 16)
(32 filters → 64 filters)             │
       │                               │
MaxPooling (1200 → 75)                 │
       │                               │
LSTM (32 units, cuDNN)                 │
       │                               │
Dense (64) ──────── Concatenate ───────┘
                         │
                    Dense (32)
                         │
                    Sigmoid output
```

**Rama A — CNN + LSTM:** Captures local morphology (QRS complex, ST elevation) via residual convolutional blocks, then temporal dependencies via LSTM.

**Rama B — statistical features:** 18 hand-crafted features per signal (RMS, kurtosis, skewness, zero-crossing rate, ST-band energy, QRS-band energy) for each of the 3 leads. These features have high discriminative power for Brugada: zero-crossing rate difference is ~67% between classes.

---

## Data Augmentation

8 techniques applied randomly (2–4 per signal) to the training set only:

| Technique | Simulates |
|-----------|-----------|
| Gaussian noise | Equipment electrical noise |
| Temporal shift | Different cycle start point |
| Amplitude scaling ±15% | Electrode impedance variation |
| Baseline wander (0.1–0.4 Hz) | Respiratory artifact |
| V3 polarity inversion | Inverted electrode placement |
| Temporal stretching ±8% | Heart rate variation |
| MixUp | Linear combination of same-class signals |
| CutMix | Segment replacement between same-class patients |

Brugada class augmented ×12, control class ×2. Val and test sets use **real signals only**.

---

## Quickstart

**1. Install dependencies**
```bash
conda create -n brugada_gpu python=3.10 -y
conda activate brugada_gpu
pip install tensorflow[and-cuda]==2.16.1
pip install -r requirements.txt
```

**2. Download dataset**
```bash
wget -r -N -c -np https://physionet.org/files/brugada-ecg-huca/1.0.0/ -P brugada_data/
```

**3. Run pipeline**
```bash
cd FASE1
python fase1_generar_dataset.py      # → X_data.npy, y_labels.npy

cd ../FASE2
python fase2_balanceo_v3.py          # → X_train/val/test.npy + feat stats

python fase3_modelo_v3.py            # → trains v3.1, saves checkpoint
python fase3_modelo_v33_finetune.py  # → fine-tunes to v3.3 (best model)
```

**4. GPU acceleration (recommended)**

The LSTM uses `recurrent_dropout=0` to enable the cuDNN fast path. On a laptop GPU (RTX 3050, 4GB) training takes ~5 minutes for 200 epochs with mixed float16 precision.

---

## Key Findings

- **Val set contamination** is a critical pitfall with small medical datasets. The val set must be split *before* augmentation — otherwise augmented synthetic samples leak into validation, producing artificially perfect val metrics and misleading EarlyStopping.
- **recurrent_dropout > 0 disables cuDNN** in TF/Keras. Setting it to 0 and using an explicit Dropout layer after the LSTM achieves the same regularization effect with ~9× speedup on GPU.
- **Threshold optimization** over the ROC curve with a clinical constraint (Recall ≥ 0.85) is more informative than accuracy. The optimal threshold was 0.525, not the default 0.5.
- **Label smoothing (ε=0.05)** improved calibration with only 61 real Brugada training samples, preventing overconfident predictions.

---

## Roadmap

- [ ] Expand to PTB-XL dataset for transfer learning and validation on a larger population
- [ ] Explainability: GRAD-CAM on temporal signals to identify which ECG regions drive predictions
- [ ] Clinical validation with cardiologists

---

## Citation

Dataset:
> Martínez-Sellés M, et al. "Brugada ECG Database from Hospital Universitario Central de Asturias (HUCA)." PhysioNet (2021). https://doi.org/10.13026/g8sm-1m65

---

## License

Code: MIT License. Dataset: PhysioNet Credentialed Health Data License — requires credentialed access at physionet.org.
