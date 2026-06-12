# Brugada Syndrome Detection from 12-Lead ECG

Deep learning system to detect **Brugada Syndrome** — a rare, genetically-driven cardiac condition that causes sudden cardiac death in otherwise healthy young adults — from standard 12-lead ECG recordings, using **transfer learning** from the large public PTB-XL dataset.

> ⚠️ **Research project only.** This system is not a medical device and must not be used for clinical diagnosis.

---

## Overview

The core engineering challenge is **data scarcity**: the specialised Brugada dataset has only 363 patients (76 positive). To overcome this, the project pre-trains a convolutional network on **PTB-XL** (20,413 clinical ECGs) to learn general ECG morphology, then fine-tunes it on the Brugada data.

The best model (**v4.1**, CNN + LSTM + transfer learning) reaches an **AUC-ROC of 0.987** with only 2 false positives and 1 false negative on the held-out test set — clearly outperforming the equivalent model trained on Brugada data alone (AUC 0.944).

---

## Results

| Model | Transfer Learning | AUC-ROC | Recall | Precision | FP | FN |
|-------|:----------------:|:-------:|:------:|:---------:|:--:|:--:|
| CNN Base (v1) | No | — | 0.933 | 0.424 | 19 | 1 |
| CNN-Res-LSTM (v2) | No | 0.953 | 0.867 | 0.565 | 11 | 2 |
| CNN-Dual (v3.1) | No | 0.944 | 0.867 | 0.765 | 4 | 2 |
| CNN-Dual fine-tune (v3.3) | No | 0.912 | 0.933 | 0.737 | 5 | 1 |
| v4 TL + GAP | **Yes** | 0.954 | 0.933 | 0.667 | 7 | 1 |
| **v4.1 TL + LSTM** ⭐ | **Yes** | **0.987** | **0.933** | **0.875** | **2** | **1** |
| **v4.2 TL + GAP** | **Yes** | 0.978 | **1.000** | 0.790 | 4 | **0** |

*Test set: 73 patients (58 controls, 15 Brugada) — never seen during training.*

Clinical priority is **Recall** (sensitivity): missing a Brugada patient is the most dangerous error. Two final models are reported as a complementary **cascade**:
- **v4.1** — best AUC, fewest false positives, more interpretable (first screener / confirmer).
- **v4.2** — Recall = 1.0, maximum sensitivity (safety net, zero missed positives).

The trained model weights are available on the [Releases page](https://github.com/dfcordero/brugada-syndrome-detection/releases). See [`models/README.md`](models/README.md) for the full model catalog.

---

## Clinical Background

Brugada Syndrome is an inherited disorder of the heart's sodium ion channels. Its diagnostic ECG signature — a characteristic *coved* ST-segment elevation — is only reliably visible in leads **V1, V2 and sometimes V3**, because these electrodes sit directly over the right ventricular outflow tract. The entire pipeline therefore restricts its input to these three leads, matching how a cardiologist reads for Brugada.

---

## Datasets

**Brugada-HUCA** (target) — 363 patients from Hospital Universitario Central de Asturias (76 Brugada positive, 287 controls), 12-lead ECG at 100 Hz, 12-second recordings. Only leads V1, V2, V3 are used → input shape `(1200, 3)`.

**PTB-XL** (pre-training) — 20,413 usable clinical ECGs at 100 Hz, multi-label across 5 superclasses (NORM, MI, STTC, CD, HYP) → input shape `(1000, 3)`.

The differing signal lengths (1200 vs 1000) are reconciled with **Global Average Pooling**, which makes the network agnostic to input length.

> Datasets are **not** included in this repository (licensing + size). See [Quickstart](#quickstart) for download instructions.

---

## Architecture

The model uses **dual inputs** processed in parallel:

```
Signal (1200, 3)              Statistical features (18,)
       │                               │
CNN Residual Blocks            MLP (Dense 32 → 16)
(32 filters → 64 filters)             │
       │                               │
MaxPooling                             │
       │                               │
LSTM (32 units, cuDNN)                 │
       │                               │
Dense (64) ──────── Concatenate ───────┘
                         │
                    Dense (32)
                         │
                    Sigmoid output
```

**Branch A — CNN + LSTM:** Captures local morphology (QRS complex, ST elevation) via residual convolutional blocks, then temporal dependencies across heartbeats via LSTM. The convolutional layers are initialised with weights pre-trained on PTB-XL.

**Branch B — statistical features:** 18 hand-crafted features per signal (RMS, kurtosis, skewness, zero-crossing rate, ST-band energy, QRS-band energy) for each of the 3 leads. These features have high discriminative power for Brugada: zero-crossing rate difference is ~67% between classes.

---

## Transfer Learning

The key to overcoming data scarcity. A residual CNN is first trained from scratch on 20,413 PTB-XL ECGs to classify 5 superclasses (macro AUC 0.865), learning general ECG morphology. Those weights are then transferred by layer name to the Brugada model and fine-tuned with **gradual unfreezing**:

1. **Phase B1** — freeze the early convolutional layers, train only the new classification head (LR 1e-4).
2. **Phase B2** — unfreeze the whole network and fine-tune with a very low learning rate (LR 1e-5) to specialise on Brugada without forgetting general ECG knowledge.

The effect is measurable: the same architecture jumps from AUC 0.944 (no transfer) to 0.987 (with transfer).

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

Brugada class augmented ×12, control class ×2. **Critically, the train/val/test split happens *before* augmentation** — val and test sets contain real signals only, preventing data leakage.

---

## Project Structure

```
brugada-syndrome-detection/
├── FASE1/                      # Raw data loading + preprocessing (notch + bandpass)
├── FASE2/                      # Baseline models (Brugada only), augmentation, data prep
├── FASE3_PTB/                  # Transfer learning pipeline
│   ├── fase3A_*_pretrain_ptbxl.py     # Phase A: PTB-XL pre-training
│   ├── fase3B_finetune.py             # Phase B: transfer + fine-tuning
│   ├── fase3B_comparativa.py          # v4.1 vs v4.2 comparison
│   ├── fase3C_gradcam.py              # Phase C: explainability (Grad-CAM)
│   ├── fase3D_validacion_ood.py       # Phase D: out-of-distribution testing
│   ├── fase3F_pretrain_6labels.py     # Phase F: 6-class differential (in progress)
│   └── fase3G_brugada_mcdropout.py    # Phase G: uncertainty / MC Dropout (in progress)
├── models/                     # Model catalog (README) — weights live in Releases
├── results/                    # Result figures (PNG)
├── .gitignore
├── requirements.txt
└── README.md
```

> Trained models (`.keras`) and data arrays (`.npy`) are git-ignored. Download the final models from [Releases](https://github.com/dfcordero/brugada-syndrome-detection/releases), or run the scripts to regenerate them.

---

## Quickstart

**1. Install dependencies**
```bash
conda create -n brugada_gpu python=3.10 -y
conda activate brugada_gpu
pip install tensorflow[and-cuda]==2.16.1
pip install -r requirements.txt
```

**2. Download PTB-XL** (via Kaggle API)
```bash
kaggle datasets download -d khyeh0719/ptb-xl-dataset --unzip
```

**3. Run the pipeline**
```bash
# Baseline (Brugada only)
cd FASE1
python fase1_generar_dataset.py      # → X_data.npy, y_labels.npy
cd ../FASE2
python fase2_balanceo_v3.py          # → X_train/val/test.npy + feature stats

# Transfer learning (from inside FASE3_PTB/)
cd ../FASE3_PTB
python fase3A_3_pretrain_ptbxl.py    # pre-train on PTB-XL
python fase3B_finetune.py            # transfer learning to Brugada → v4.1, v4.2
python fase3C_gradcam.py             # explainability
python fase3D_validacion_ood.py      # out-of-distribution testing
```

**4. GPU acceleration (recommended)**

The LSTM uses `recurrent_dropout=0` to enable the cuDNN fast path. On a laptop GPU (RTX 3050, 4 GB) training takes ~5 minutes with mixed float16 precision.

---

## Key Findings

- **Transfer learning is what makes it work.** With only ~290 training samples, a complex CNN+LSTM overfits. Pre-training on 20,413 PTB-XL ECGs gives the convolutional layers a sensible starting point, lifting the same architecture from AUC 0.944 to 0.987.
- **Val set contamination** is a critical pitfall with small medical datasets. The split must happen *before* augmentation — otherwise synthetic samples leak into validation, producing artificially perfect val metrics and misleading EarlyStopping.
- **recurrent_dropout > 0 disables cuDNN** in TF/Keras. Setting it to 0 and using an explicit Dropout layer after the LSTM achieves the same regularization with ~9× GPU speedup.
- **Threshold optimization** over the ROC curve with a clinical constraint (Recall ≥ 0.85) is more informative than accuracy.
- **Label smoothing (ε=0.05)** improved calibration with so few real Brugada training samples, preventing overconfident predictions.
- **Explainability check (Grad-CAM):** v4.1 focuses on the localised ST region of early beats, matching the clinical diagnostic criteria for Brugada.

---

## Roadmap

- [ ] Differential diagnosis: Brugada as a 6th label (vs other pathologies, not just healthy controls)
- [ ] Hard-negative mining (RBBB / MI cases as difficult negatives)
- [ ] Stratified 5-fold cross-validation + bootstrap CIs to validate the 0.987 AUC
- [ ] Quantitative Grad-CAM validation (activation inside the ST window)
- [ ] TensorFlow Lite Int8 quantisation for edge deployment
- [ ] Streamlit web app for interactive inference
- [ ] Clinical validation with cardiologists

---

## Citation

Dataset:
> Martínez-Sellés M, et al. "Brugada ECG Database from Hospital Universitario Central de Asturias (HUCA)." PhysioNet (2021). https://doi.org/10.13026/g8sm-1m65

PTB-XL:
> Wagner P, et al. "PTB-XL, a large publicly available electrocardiography dataset." Scientific Data 7, 154 (2020). https://doi.org/10.1038/s41597-020-0495-6

---

## License

Code: MIT License. Datasets retain their original licenses (PhysioNet Credentialed Health Data License / PTB-XL under Creative Commons).

---

> ⚠️ **Disclaimer:** Research and educational use only. Not a certified medical device; not for clinical decision-making.
