"""
================================================================================
  DETECCIÓN SÍNDROME DE BRUGADA — FASE 2 v3: AUGMENTATION + VAL LIMPIO
================================================================================
  Mejoras respecto a v2:
    · FIX CRÍTICO: Val set separado ANTES del augmentation (solo datos reales)
      → Elimina la inestabilidad de val_loss que vimos en las gráficas de v2
    · MixUp augmentation: mezcla ponderada señal+etiqueta entre pares
      → Regularización implícita muy efectiva en bioseñales pequeñas
    · Brugada x12 (antes x10) → más diversidad sintética
    · Guarda X_val/y_val por separado para validation_data en model.fit()
================================================================================
"""

import numpy as np
import os
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
RUTA_FASE1       = '../'
RANDOM_SEED      = 42
TEST_SIZE        = 0.20
VAL_SIZE         = 0.15   # porcentaje del TOTAL — se separa antes de augmentar
BRUGADA_AUG_MULT = 12
CONTROL_AUG_MULT = 2
N_MIXUP          = 200    # muestras MixUp adicionales

np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. TÉCNICAS DE AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────

def add_gaussian_noise(signal, noise_level=0.012):
    return signal + np.random.normal(0, noise_level, signal.shape)

def temporal_shift(signal, shift_max=25):
    return np.roll(signal, np.random.randint(-shift_max, shift_max), axis=0)

def amplitude_scaling(signal, scale_range=(0.85, 1.15)):
    return signal * np.random.uniform(*scale_range)

def baseline_wander(signal, fs=100, max_amplitude=0.08):
    n = signal.shape[0]
    t = np.linspace(0, n / fs, n)
    freq = np.random.uniform(0.1, 0.4)
    wander = np.random.uniform(0.01, max_amplitude) * np.sin(
        2 * np.pi * freq * t + np.random.uniform(0, 2 * np.pi))
    return signal + wander[:, np.newaxis]

def amplitude_inversion(signal):
    aug = signal.copy()
    if np.random.random() > 0.5:
        aug[:, 2] = -aug[:, 2]
    return aug

def temporal_stretching(signal, stretch_range=(0.92, 1.08)):
    n = signal.shape[0]
    new_len = int(n * np.random.uniform(*stretch_range))
    x_orig = np.linspace(0, 1, n)
    x_new  = np.linspace(0, 1, new_len)
    out = np.zeros_like(signal)
    for ch in range(signal.shape[1]):
        out[:, ch] = np.interp(x_orig, x_new, np.interp(x_new, x_orig, signal[:, ch]))
    return out

# NUEVA TÉCNICA: voltaje cutout — pone a cero una ventana temporal aleatoria
# Simula segmentos del ECG con artefacto de movimiento o electrodo suelto
def voltage_cutout(signal, max_width=60):
    """Pone a cero una ventana aleatoria (hasta 60 muestras = 0.6 s)."""
    aug = signal.copy()
    width = np.random.randint(10, max_width)
    start = np.random.randint(0, signal.shape[0] - width)
    aug[start:start + width, :] = 0.0
    return aug

AUGMENTATION_POOL = [
    add_gaussian_noise, temporal_shift, amplitude_scaling,
    baseline_wander, amplitude_inversion, temporal_stretching,
    voltage_cutout,
]

def augment_ecg(signal):
    """Aplica 2–4 técnicas aleatorias del pool."""
    techs = np.random.choice(AUGMENTATION_POOL,
                              size=np.random.randint(2, 5), replace=False)
    aug = signal.copy()
    for t in techs:
        aug = t(aug)
    return aug


# ─────────────────────────────────────────────────────────────────────────────
# 2. MIXUP AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# Mezcla dos señales con lambda ~ Beta(alpha, alpha).
# Clave: las etiquetas resultantes son BLANDAS (ej. 0.62, 0.38).
# Esto obliga al modelo a aprender una frontera de decisión suave y 
# generalizable en lugar de hiperespacios rígidos que sobreajustan.
# Validado en ECG: Gorodkin et al. 2021, Wang et al. 2022.

def mixup_batch(X, y, n_samples, alpha=0.4):
    """
    Args:
        X        : array (N, 1200, 3)
        y        : array (N,) con valores 0.0 o 1.0
        n_samples: cuántas muestras generar
        alpha    : parámetro Beta — 0.4 → mezclas moderadas
    Returns:
        X_mix (n_samples, 1200, 3), y_mix (n_samples,) etiquetas blandas
    """
    n = len(X)
    X_mix = np.zeros((n_samples,) + X.shape[1:], dtype=np.float32)
    y_mix = np.zeros(n_samples, dtype=np.float32)
    for i in range(n_samples):
        a, b = np.random.randint(0, n, size=2)
        lam  = np.random.beta(alpha, alpha)
        X_mix[i] = lam * X[a] + (1 - lam) * X[b]
        y_mix[i] = lam * float(y[a]) + (1 - lam) * float(y[b])
    return X_mix, y_mix


# ─────────────────────────────────────────────────────────────────────────────
# 3. CARGA DE DATOS ORIGINALES
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("  FASE 2 v3 — AUGMENTATION + VAL LIMPIO + MIXUP")
print("=" * 60)

try:
    X = np.load(os.path.join(RUTA_FASE1, 'X_data.npy'))
    y = np.load(os.path.join(RUTA_FASE1, 'y_labels.npy'))
    print(f"  ✓ X: {X.shape}  Brugada={int((y==1).sum())}  Control={int((y==0).sum())}")
except FileNotFoundError:
    print("  [ERROR] Ajusta RUTA_FASE1 al inicio del script.")
    exit()


# ─────────────────────────────────────────────────────────────────────────────
# 4. TRIPLE SPLIT ESTRATIFICADO — ORDEN IMPORTA
# ─────────────────────────────────────────────────────────────────────────────
# Paso 1: separar test (20%)
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y)

# Paso 2: separar val del trainval (15% del total ≈ 18.75% del trainval)
val_frac = VAL_SIZE / (1 - TEST_SIZE)
X_train_real, X_val, y_train_real, y_val = train_test_split(
    X_trainval, y_trainval,
    test_size=val_frac, random_state=RANDOM_SEED, stratify=y_trainval)

print(f"\n  SPLITS REALES (sin augmentation):")
print(f"  Train → C:{int((y_train_real==0).sum())}  B:{int((y_train_real==1).sum())}  Total:{len(X_train_real)}")
print(f"  Val   → C:{int((y_val==0).sum())}  B:{int((y_val==1).sum())}  Total:{len(X_val)}")
print(f"  Test  → C:{int((y_test==0).sum())}  B:{int((y_test==1).sum())}  Total:{len(X_test)}")
print(f"\n  Val y Test son 100% datos reales — nunca aumentados")


# ─────────────────────────────────────────────────────────────────────────────
# 5. AUGMENTATION SOLO EN TRAIN
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n  Augmentando Brugada x{BRUGADA_AUG_MULT}, Control x{CONTROL_AUG_MULT}...")

def augment_class(X_class, label, mult):
    X_out, y_out = [], []
    for sig in X_class:
        for _ in range(mult):
            X_out.append(augment_ecg(sig))
            y_out.append(float(label))
    return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.float32)

X_aug_b, y_aug_b = augment_class(X_train_real[y_train_real==1], 1, BRUGADA_AUG_MULT)
X_aug_c, y_aug_c = augment_class(X_train_real[y_train_real==0], 0, CONTROL_AUG_MULT)

print(f"  ✓ Brugada sintéticas : {len(X_aug_b)}")
print(f"  ✓ Control sintéticas : {len(X_aug_c)}")

# MixUp — sobre train real (antes de mezclar con aug)
print(f"\n  Generando {N_MIXUP} muestras MixUp...")
X_mix, y_mix = mixup_batch(
    X_train_real.astype(np.float32),
    y_train_real.astype(np.float32),
    n_samples=N_MIXUP, alpha=0.4)
print(f"  ✓ Etiquetas blandas — rango [{y_mix.min():.2f}, {y_mix.max():.2f}]")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CONCATENAR Y MEZCLAR
# ─────────────────────────────────────────────────────────────────────────────
X_train_final = np.concatenate([
    X_train_real.astype(np.float32),
    X_aug_b, X_aug_c, X_mix
])
y_train_final = np.concatenate([
    y_train_real.astype(np.float32),
    y_aug_b, y_aug_c, y_mix
])

idx = np.random.permutation(len(X_train_final))
X_train_final = X_train_final[idx]
y_train_final = y_train_final[idx]

print(f"\n  TRAIN FINAL: {len(X_train_final)} muestras")
hard = (y_train_final == 0) | (y_train_final == 1)
print(f"  Hard labels: {hard.sum()}  |  Soft (MixUp): {(~hard).sum()}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
np.save('X_train.npy', X_train_final)
np.save('y_train.npy', y_train_final)
np.save('X_val.npy',   X_val.astype(np.float32))
np.save('y_val.npy',   y_val.astype(np.float32))
np.save('X_test.npy',  X_test.astype(np.float32))
np.save('y_test.npy',  y_test.astype(np.float32))

print(f"""
  ✓ Archivos guardados:
    X_train.npy  {X_train_final.shape}
    X_val.npy    {X_val.shape}      ← solo datos reales
    X_test.npy   {X_test.shape}     ← solo datos reales

╔══════════════════════════════════════════════════╗
║  FASE 2 v3 — RESUMEN                             ║
╠══════════════════════════════════════════════════╣
║  Train total  : {len(X_train_final):5d} (aug + mixup)        ║
║  Val real     : {len(X_val):5d} (sin sintéticas)         ║
║  Test real    : {len(X_test):5d} (sin sintéticas)         ║
╚══════════════════════════════════════════════════╝
  → Ejecuta a continuación: fase3_modelo_v3.py
""")
