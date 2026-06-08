"""
fase3D_validacion_ood.py
========================
Validación Out-Of-Distribution (OOD) para modelos de detección de Brugada v4.1 y v4.2.

Los modelos fueron entrenados SOLO con pacientes Control vs Brugada.
Este script evalúa su comportamiento ante patologías PTB-XL nunca vistas:
  NORM, MI, STTC, CD, HYP  (grupos puros: una sola superclase activa)

Métricas: tasa de falsos positivos (FP_rate), prob_media, distribución de probs.

Ejecutar desde: ~/proyectos/brugada_syndrom/FASE3_PTB/
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import kurtosis, skew
from pathlib import Path

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
LABELS_CSV = BASE_DIR / "ptb_xl_data/processed/labels.csv"
RECORDS_DIR = BASE_DIR / "ptb_xl_data/processed/records"
MODEL_V41  = BASE_DIR / "modelo_brugada_v41.keras"
MODEL_V42  = BASE_DIR / "modelo_brugada_v42.keras"
UMBRAL_V41 = BASE_DIR / "umbral_v41.npy"
UMBRAL_V42 = BASE_DIR / "umbral_v42.npy"
FEAT_MEAN  = BASE_DIR / "../FASE2/feat_mean.npy"
FEAT_STD   = BASE_DIR / "../FASE2/feat_std.npy"
OUT_FIG    = BASE_DIR / "fase3D_ood_results.png"

# Umbrales por defecto (overrideados por archivos .npy si existen)
DEFAULT_THR_V41 = 0.825
DEFAULT_THR_V42 = 0.525

# Superclases a evaluar (en orden de figura)
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
MAX_PER_CLASS = 1000   # cap para no saturar memoria; None = sin límite


# ─── Utilidades ───────────────────────────────────────────────────────────────
def load_threshold(path, default):
    if Path(path).exists():
        val = float(np.load(path))
        print(f"  Umbral cargado de {Path(path).name}: {val:.4f}")
        return val
    print(f"  Umbral no encontrado en {Path(path).name}, usando default={default}")
    return default


def extract_statistical_features(X, fs=100):
    """
    Extrae 18 features estadísticos de un array de señales.
    X: ndarray shape (N, n_samp, 3)
    Returns: ndarray shape (N, 18)
    Orden: [RMS, kurtosis, skewness, ZCR, E_ST, E_QRS] × [V1, V2, V3]
    """
    N, n_samp, _ = X.shape
    freqs = np.fft.rfftfreq(n_samp, 1.0 / fs)
    feats = np.zeros((N, 18), dtype=np.float32)

    for i in range(N):
        for ch in range(3):
            sig  = X[i, :, ch].astype(np.float64)
            base = ch * 6
            mu   = sig.mean()
            std  = sig.std() + 1e-8

            # RMS
            feats[i, base + 0] = np.sqrt(np.mean(sig ** 2))
            # Kurtosis (4.º momento normalizado)
            feats[i, base + 1] = np.mean(((sig - mu) / std) ** 4)
            # Skewness (3.er momento normalizado)
            feats[i, base + 2] = np.mean(((sig - mu) / std) ** 3)
            # Zero Crossing Rate
            feats[i, base + 3] = ((sig[:-1] * sig[1:]) < 0).sum() / n_samp
            # Energía espectral
            fft_m = np.abs(np.fft.rfft(sig))
            e_tot = fft_m.sum() + 1e-8
            feats[i, base + 4] = fft_m[(freqs >= 0.5) & (freqs <= 5)].sum() / e_tot  # E_ST
            feats[i, base + 5] = fft_m[(freqs >= 5) & (freqs <= 40)].sum() / e_tot   # E_QRS

    return feats


def load_group(df_group, records_dir, max_n=None, fs=100):
    """
    Carga las señales .npy de un grupo (DataFrame con columna ecg_id).
    Devuelve X (N, 1000, 3) float32.
    """
    ids = df_group["ecg_id"].tolist()
    if max_n is not None:
        rng = np.random.default_rng(42)
        ids = rng.choice(ids, size=min(max_n, len(ids)), replace=False).tolist()

    signals = []
    missing = 0
    for eid in ids:
        fpath = Path(records_dir) / f"{eid}.npy"
        if not fpath.exists():
            missing += 1
            continue
        sig = np.load(fpath).astype(np.float32)
        if sig.ndim == 1:
            sig = sig.reshape(-1, 1)
        if sig.shape[0] != 1000 or sig.shape[1] < 3:
            missing += 1
            continue
        signals.append(sig[:, :3])

    if missing:
        print(f"    Advertencia: {missing} registros faltantes o con shape incorrecto.")
    if not signals:
        return np.empty((0, 1000, 3), dtype=np.float32)

    return np.stack(signals, axis=0)


def predict_model(model, X_sig, X_feat):
    """
    Inferencia dual-input: [señal, features].
    Devuelve array 1D de probabilidades de Brugada.
    """
    preds = model.predict([X_sig, X_feat], batch_size=64, verbose=0)
    return preds.ravel()


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    import tensorflow as tf

    print("=" * 60)
    print("  Validación OOD — Síndrome de Brugada")
    print("=" * 60)

    # ── 1. Cargar etiquetas ──────────────────────────────────────────────────
    print("\n[1] Cargando labels.csv …")
    df = pd.read_csv(LABELS_CSV)
    print(f"    Total registros: {len(df)}")
    print(f"    Columnas: {list(df.columns)}")

    # Grupos puros: exactamente UNA superclase = 1, resto = 0
    groups = {}
    for sc in SUPERCLASSES:
        mask = (df[sc] == 1)
        for other in SUPERCLASSES:
            if other != sc:
                mask &= (df[other] == 0)
        groups[sc] = df[mask].reset_index(drop=True)
        print(f"    {sc} puro: {len(groups[sc])} registros")

    # ── 2. Cargar normalizadores de features ─────────────────────────────────
    print("\n[2] Cargando normalizadores de features …")
    feat_mean = np.load(FEAT_MEAN).astype(np.float32)
    feat_std  = np.load(FEAT_STD).astype(np.float32)
    print(f"    feat_mean shape: {feat_mean.shape}, feat_std shape: {feat_std.shape}")

    # ── 3. Cargar modelos y umbrales ─────────────────────────────────────────
    # Función de pérdida personalizada (label smoothing ε=0.05)
    def bce_label_smoothing(epsilon=0.05):
        def loss_fn(y_true, y_pred):
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
            y_smooth = y_true * (1.0 - epsilon) + epsilon / 2.0
            return tf.reduce_mean(
                -(y_smooth * tf.math.log(y_pred) +
                  (1 - y_smooth) * tf.math.log(1 - y_pred))
            )
        loss_fn.__name__ = f'bce_smooth_{epsilon}'
        return loss_fn

    custom_objs = {'bce_smooth_0.05': bce_label_smoothing(0.05)}

    print("\n[3] Cargando modelos …")
    print(f"    Cargando v4.1 desde {MODEL_V41.name} …")
    model_v41 = tf.keras.models.load_model(str(MODEL_V41), custom_objects=custom_objs)
    print(f"    Cargando v4.2 desde {MODEL_V42.name} …")
    model_v42 = tf.keras.models.load_model(str(MODEL_V42), custom_objects=custom_objs)

    thr_v41 = load_threshold(UMBRAL_V41, DEFAULT_THR_V41)
    thr_v42 = load_threshold(UMBRAL_V42, DEFAULT_THR_V42)

    # Resumen de shapes de entrada
    print(f"\n    v4.1 inputs: {[inp.shape for inp in model_v41.inputs]}")
    print(f"    v4.2 inputs: {[inp.shape for inp in model_v42.inputs]}")

    # ── 4. Inferencia por grupo ───────────────────────────────────────────────
    print("\n[4] Inferencia por grupo de patología …")

    results = {}

    for sc in SUPERCLASSES:
        print(f"\n  ── {sc} ──")
        df_grp = groups[sc]
        if len(df_grp) == 0:
            print("    Sin registros, saltando.")
            results[sc] = None
            continue

        # Cargar señales
        print(f"    Cargando señales (max={MAX_PER_CLASS}) …")
        X = load_group(df_grp, RECORDS_DIR, max_n=MAX_PER_CLASS)
        n = len(X)
        if n == 0:
            print("    Sin señales válidas.")
            results[sc] = None
            continue
        print(f"    Señales cargadas: {n} × {X.shape[1:]}  (dtype={X.dtype})")

        # Extraer y normalizar features
        print("    Extrayendo 18 features estadísticos …")
        feats_raw = extract_statistical_features(X, fs=100)
        feats_norm = (feats_raw - feat_mean) / (feat_std + 1e-8)
        feats_norm = feats_norm.astype(np.float32)

        # v4.1: acepta (N, 1000, 3)
        print("    Predicción v4.1 …")
        probs_v41 = predict_model(model_v41, X, feats_norm)

        # v4.2: señal completa tal cual (1000, 3); GAP es length-agnostic
        print("    Predicción v4.2 …")
        probs_v42 = predict_model(model_v42, X, feats_norm)

        fp_v41 = (probs_v41 >= thr_v41).mean() * 100
        fp_v42 = (probs_v42 >= thr_v42).mean() * 100

        results[sc] = {
            "n":       n,
            "probs_v41": probs_v41,
            "probs_v42": probs_v42,
            "fp_v41":  fp_v41,
            "fp_v42":  fp_v42,
            "mean_v41": probs_v41.mean() * 100,
            "mean_v42": probs_v42.mean() * 100,
        }
        print(f"    v4.1 → FP={fp_v41:.1f}%  prob_media={probs_v41.mean()*100:.1f}%")
        print(f"    v4.2 → FP={fp_v42:.1f}%  prob_media={probs_v42.mean()*100:.1f}%")

    # ── 5. Figura de resultados ───────────────────────────────────────────────
    print("\n[5] Generando figura …")

    valid_sc = [sc for sc in SUPERCLASSES if results.get(sc) is not None]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        "Validación OOD — Modelos v4.1 (CNN+LSTM) y v4.2 (CNN+GAP)\n"
        "Comportamiento ante patologías PTB-XL nunca vistas durante entrenamiento",
        fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.45,
                           top=0.92, bottom=0.08)

    colors_v41 = "#E05252"
    colors_v42 = "#5278E0"
    x_pos = np.arange(len(valid_sc))
    bar_w = 0.35

    # ── 5a. Barplot FP rate ────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    fp41 = [results[sc]["fp_v41"] for sc in valid_sc]
    fp42 = [results[sc]["fp_v42"] for sc in valid_sc]

    b1 = ax1.bar(x_pos - bar_w / 2, fp41, bar_w,
                 label=f"v4.1 (thr={thr_v41:.3f})", color=colors_v41, alpha=0.85)
    b2 = ax1.bar(x_pos + bar_w / 2, fp42, bar_w,
                 label=f"v4.2 (thr={thr_v42:.3f})", color=colors_v42, alpha=0.85)

    # Etiquetas sobre barras
    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color=colors_v41)
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                 f"{h:.1f}%", ha="center", va="bottom", fontsize=8, color=colors_v42)

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(valid_sc, fontsize=11)
    ax1.set_ylabel("Tasa de Falsos Positivos (%)", fontsize=10)
    ax1.set_title("Tasa de Falsos Positivos por Patología OOD", fontsize=11, fontweight="bold")
    ax1.set_ylim(0, max(max(fp41 + fp42) * 1.25, 5))
    ax1.axhline(5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7, label="5% referencia")
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_facecolor("#F8F8F8")

    # ── 5b. Boxplot distribución de probabilidades ──────────────────────────
    ax2 = fig.add_subplot(gs[1])

    bp_data_v41 = [results[sc]["probs_v41"] * 100 for sc in valid_sc]
    bp_data_v42 = [results[sc]["probs_v42"] * 100 for sc in valid_sc]

    # Intercalar boxplots v41 / v42
    positions_v41 = x_pos * 2.5 - 0.5
    positions_v42 = x_pos * 2.5 + 0.5

    bp1 = ax2.boxplot(bp_data_v41, positions=positions_v41, widths=0.7,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=colors_v41, alpha=0.5),
                      medianprops=dict(color="darkred", linewidth=2),
                      whiskerprops=dict(color=colors_v41),
                      capprops=dict(color=colors_v41))
    bp2 = ax2.boxplot(bp_data_v42, positions=positions_v42, widths=0.7,
                      patch_artist=True, showfliers=False,
                      boxprops=dict(facecolor=colors_v42, alpha=0.5),
                      medianprops=dict(color="darkblue", linewidth=2),
                      whiskerprops=dict(color=colors_v42),
                      capprops=dict(color=colors_v42))

    # Línea de umbral aproximado (media de ambos)
    thr_mid = (thr_v41 + thr_v42) / 2 * 100
    ax2.axhline(thr_v41 * 100, color=colors_v41, linestyle="--", linewidth=0.9,
                alpha=0.8, label=f"Umbral v4.1 ({thr_v41:.3f})")
    ax2.axhline(thr_v42 * 100, color=colors_v42, linestyle="--", linewidth=0.9,
                alpha=0.8, label=f"Umbral v4.2 ({thr_v42:.3f})")

    xtick_pos = (positions_v41 + positions_v42) / 2
    ax2.set_xticks(xtick_pos)
    ax2.set_xticklabels(valid_sc, fontsize=11)
    ax2.set_ylabel("P(Brugada) (%)", fontsize=10)
    ax2.set_title("Distribución de Probabilidad de Brugada por Patología OOD", fontsize=11, fontweight="bold")
    ax2.set_ylim(-2, 105)

    # Leyenda manual
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors_v41, alpha=0.6, label=f"v4.1 (thr={thr_v41:.3f})"),
        Patch(facecolor=colors_v42, alpha=0.6, label=f"v4.2 (thr={thr_v42:.3f})"),
    ]
    ax2.legend(handles=legend_elements, fontsize=9)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_facecolor("#F8F8F8")

    # ── 5c. Tabla resumen ────────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")

    table_data = []
    col_labels = [
        "Patología", "N",
        "FP v4.1 (%)", "P̄(Brugada) v4.1 (%)",
        "FP v4.2 (%)", "P̄(Brugada) v4.2 (%)"
    ]

    for sc in valid_sc:
        r = results[sc]
        table_data.append([
            sc,
            str(r["n"]),
            f"{r['fp_v41']:.1f}",
            f"{r['mean_v41']:.1f}",
            f"{r['fp_v42']:.1f}",
            f"{r['mean_v42']:.1f}",
        ])

    table = ax3.table(
        cellText=table_data,
        colLabels=col_labels,
        loc="center",
        cellLoc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Estilo de cabecera
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor("#2C3E50")
        table[(0, j)].set_text_props(color="white", fontweight="bold")

    # Estilo de filas alternas
    for i in range(1, len(table_data) + 1):
        bg = "#EBF5FB" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(bg)

    # Resaltar FP altas (> 10%)
    for i, sc in enumerate(valid_sc, start=1):
        r = results[sc]
        if r["fp_v41"] > 10:
            table[(i, 2)].set_facecolor("#FADBD8")
            table[(i, 2)].set_text_props(fontweight="bold")
        if r["fp_v42"] > 10:
            table[(i, 4)].set_facecolor("#FADBD8")
            table[(i, 4)].set_text_props(fontweight="bold")

    ax3.set_title("Tabla Resumen OOD", fontsize=11, fontweight="bold", pad=10)

    plt.savefig(OUT_FIG, dpi=150, bbox_inches="tight")
    print(f"    Figura guardada en: {OUT_FIG}")

    # ── 6. Resumen textual ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  RESUMEN FINAL OOD")
    print("=" * 60)
    print(f"\n  {'Patología':<8} {'N':>6}  {'FP v4.1':>9}  {'P̄ v4.1':>9}  {'FP v4.2':>9}  {'P̄ v4.2':>9}")
    print("  " + "-" * 58)
    for sc in valid_sc:
        r = results[sc]
        print(f"  {sc:<8} {r['n']:>6}  {r['fp_v41']:>8.1f}%  {r['mean_v41']:>8.1f}%  {r['fp_v42']:>8.1f}%  {r['mean_v42']:>8.1f}%")

    print("\n  Interpretación:")
    print("  - FP rate alta = modelo confunde esa patología con Brugada.")
    print("  - Prob media alta = modelo asigna score elevado aunque sean controles.")
    print(f"  - Umbral v4.1={thr_v41:.3f}  Umbral v4.2={thr_v42:.3f}")
    print("\n  Script completado correctamente.")


if __name__ == "__main__":
    main()
