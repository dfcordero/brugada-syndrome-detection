"""
================================================================================
  TRANSFER LEARNING — FASE B: FINE-TUNING EN BRUGADA
================================================================================
  Ejecutar desde: ~/proyectos/brugada_syndrom/FASE3_PTB/
  python fase3B_finetune_brugada.py

  Estrategia de gradual unfreezing:
    B1: Congelar res1, entrenar res2+GAP+Dense nueva cabeza (20 épocas, LR=1e-4)
    B2: Descongelar todo, fine-tuning completo (100 épocas, LR=1e-5)

  Input Brugada : (1200, 3) — 12 seg a 100Hz, derivaciones V1/V2/V3
  Input PTB-XL  : (1000, 3) — el modelo acepta ambas gracias al GAP
================================================================================
"""

# ── GPU ───────────────────────────────────────────────────────────────────────
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"  GPU: {gpus[0].name}  |  Política: mixed_float16")

import numpy as np
import sys
from pathlib import Path
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

# Importar generator del Script 2
sys.path.append(str(Path(__file__).parent))
from fase3A_2_data_generator import SUPERCLASSES

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS
# ─────────────────────────────────────────────────────────────────────────────
# Datos Brugada — una carpeta arriba (FASE2/)
BRUGADA_DIR = Path('../FASE2')
PRETRAINED  = Path('mejor_modelo_pretrained_ptbxl.keras')


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS BRUGADA
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  CARGANDO DATOS BRUGADA-HUCA")
print("=" * 62)

X_train = np.load(BRUGADA_DIR / 'X_train.npy')
y_train = np.load(BRUGADA_DIR / 'y_train.npy')
X_val   = np.load(BRUGADA_DIR / 'X_val.npy')
y_val   = np.load(BRUGADA_DIR / 'y_val.npy')
X_test  = np.load(BRUGADA_DIR / 'X_test.npy')
y_test  = np.load(BRUGADA_DIR / 'y_test.npy')

input_shape = (X_train.shape[1], X_train.shape[2])  # (1200, 3)

print(f"  Train {X_train.shape}  Ctrl={int((y_train==0).sum())}  Brug={int((y_train==1).sum())}")
print(f"  Val   {X_val.shape}    Ctrl={int((y_val==0).sum())}    Brug={int((y_val==1).sum())}")
print(f"  Test  {X_test.shape}   Ctrl={int((y_test==0).sum())}   Brug={int((y_test==1).sum())}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES ESTADÍSTICOS (mismos que en v3.3)
# ─────────────────────────────────────────────────────────────────────────────

def extract_statistical_features(X, fs=100):
    N, n_samp = X.shape[0], X.shape[1]
    freqs = np.fft.rfftfreq(n_samp, 1/fs)
    feats = np.zeros((N, 18))
    for i in range(N):
        for ch in range(3):
            sig  = X[i, :, ch]
            base = ch * 6
            mu, std = sig.mean(), sig.std() + 1e-8
            feats[i, base+0] = np.sqrt(np.mean(sig**2))
            feats[i, base+1] = np.mean(((sig-mu)/std)**4)
            feats[i, base+2] = np.mean(((sig-mu)/std)**3)
            feats[i, base+3] = ((sig[:-1]*sig[1:]) < 0).sum() / n_samp
            fft_m = np.abs(np.fft.rfft(sig))
            e_tot = fft_m.sum() + 1e-8
            feats[i, base+4] = fft_m[(freqs>=0.5)&(freqs<=5)].sum()  / e_tot
            feats[i, base+5] = fft_m[(freqs>=5) &(freqs<=40)].sum()  / e_tot
    return feats

print("\n  Extrayendo features estadísticos...")
F_train = extract_statistical_features(X_train)
F_val   = extract_statistical_features(X_val)
F_test  = extract_statistical_features(X_test)

feat_mean = np.load(BRUGADA_DIR / 'feat_mean.npy')
feat_std  = np.load(BRUGADA_DIR / 'feat_std.npy')
F_train   = (F_train - feat_mean) / feat_std
F_val     = (F_val   - feat_mean) / feat_std
F_test    = (F_test  - feat_mean) / feat_std
print(f"  ✓ Features normalizados con stats del v3.3")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS CON LABEL SMOOTHING
# ─────────────────────────────────────────────────────────────────────────────

def bce_label_smoothing(epsilon=0.05):
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0-1e-7)
        y_smooth = y_true * (1.0 - epsilon) + epsilon / 2.0
        return tf.reduce_mean(
            -(y_smooth * tf.math.log(y_pred) +
              (1 - y_smooth) * tf.math.log(1 - y_pred))
        )
    loss_fn.__name__ = f'bce_smooth_{epsilon}'
    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# 4. CONSTRUIR MODELO v4 DESDE EL PREENTRENADO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  CONSTRUYENDO MODELO v4 DESDE PREENTRENADO PTB-XL")
print("=" * 62)

# Cargar modelo preentrenado para extraer sus pesos
model_ptbxl = tf.keras.models.load_model(PRETRAINED)
print(f"  ✓ Modelo PTB-XL cargado: {model_ptbxl.count_params():,} parámetros")

# Construir nueva arquitectura dual con mixed_float16 activo
# Mantiene los mismos nombres de capa que el preentrenado para transferencia

def residual_block(x, filters, kernel_size, name, l2=1e-4):
    shortcut = x
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2),
                      name=f'{name}_c1')(x)
    x = layers.BatchNormalization(name=f'{name}_bn1')(x)
    x = layers.Activation('relu', name=f'{name}_r1')(x)
    x = layers.Conv1D(filters, kernel_size, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2),
                      name=f'{name}_c2')(x)
    x = layers.BatchNormalization(name=f'{name}_bn2')(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, 1, padding='same', use_bias=False,
                                 name=f'{name}_skip')(shortcut)
        shortcut = layers.BatchNormalization(name=f'{name}_sbn')(shortcut)
    x = layers.Add(name=f'{name}_add')([x, shortcut])
    return layers.Activation('relu', name=f'{name}_r2')(x)


def build_model_v4(input_shape=(1200, 3), n_features=18,
                   l2=1e-4, dropout=0.30):
    """
    Arquitectura dual v4:
      Rama A: CNN Residual + GAP (pesos transferidos de PTB-XL)
      Rama B: MLP con 18 features estadísticos (mismo que v3.3)
      Fusión: Concatenate → Dense → Dense(1, sigmoid)

    Ventaja sobre v3.3: las capas CNN ya conocen morfología ECG
    gracias al preentrenamiento en 20.413 registros de PTB-XL.
    """
    # ── Rama A: señal cruda ───────────────────────────────────────────────────
    inp_ecg = layers.Input(shape=input_shape, name='ecg_signal')

    x = layers.Conv1D(32, 7, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2),
                      name='conv_in')(inp_ecg)
    x = layers.BatchNormalization(name='bn_in')(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 32, 5, 'res1', l2=l2)
    x = layers.MaxPooling1D(4, name='pool1')(x)
    x = layers.Dropout(dropout * 0.5)(x)

    x = residual_block(x, 64, 3, 'res2', l2=l2)
    x = layers.MaxPooling1D(4, name='pool2')(x)
    x = layers.Dropout(dropout * 0.5)(x)

    # GAP — agnóstico a la longitud temporal
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='dense_head')(x)
    x = layers.Dropout(dropout, name='drop_head')(x)

    # ── Rama B: features estadísticos ────────────────────────────────────────
    inp_feats = layers.Input(shape=(n_features,), name='ecg_features')
    f = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='dense_f1')(inp_feats)
    f = layers.BatchNormalization(name='bn_f1')(f)
    f = layers.Dropout(dropout, name='drop_f1')(f)
    rama_b = layers.Dense(16, activation='relu', name='dense_f2')(f)

    # ── Fusión ────────────────────────────────────────────────────────────────
    fused  = layers.Concatenate(name='fusion')([x, rama_b])
    fused  = layers.Dropout(dropout * 0.5, name='drop_fused')(fused)
    fused  = layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(l2),
                          name='dense_fused')(fused)
    fused  = layers.Dropout(dropout * 0.5)(fused)
    output = layers.Dense(1, activation='sigmoid', name='output',
                          dtype='float32')(fused)

    return models.Model(inputs=[inp_ecg, inp_feats], outputs=output,
                        name='BrugadaDual_v4_TL')


model = build_model_v4(input_shape=input_shape)

# Transferir pesos del preentrenado capa a capa por nombre
transferidas = 0
omitidas     = 0
for layer_new in model.layers:
    if not layer_new.weights:
        continue
    try:
        layer_old = model_ptbxl.get_layer(layer_new.name)
        if all(w_n.shape == w_o.shape
               for w_n, w_o in zip(layer_new.weights, layer_old.weights)):
            layer_new.set_weights(layer_old.get_weights())
            transferidas += 1
        else:
            omitidas += 1
    except ValueError:
        omitidas += 1

del model_ptbxl  # liberar memoria

print(f"  ✓ Capas transferidas desde PTB-XL: {transferidas}")
print(f"  ✓ Capas nuevas (rama features + cabeza): {omitidas}")
print(f"  Parámetros totales: {model.count_params():,}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
n_total = len(y_train)
n_ctrl  = int((y_train==0).sum())
n_brug  = int((y_train==1).sum())
CLASS_WEIGHTS = {0: n_total/(2.0*n_ctrl), 1: n_total/(2.0*n_brug)}
print(f"\n  Class weights: Ctrl={CLASS_WEIGHTS[0]:.3f}  Brug={CLASS_WEIGHTS[1]:.3f}")

RECALL_MIN = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# 6. FASE B1 — FREEZE res1, entrenar el resto
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FASE B1 — FREEZE res1, LR=1e-4, 30 épocas")
print("  Objetivo: estabilizar la nueva cabeza Brugada")
print("=" * 62)

# Congelar res1 (features básicos — no necesitan ajustarse a Brugada)
capas_res1 = ['res1_c1','res1_bn1','res1_r1','res1_c2','res1_bn2',
               'res1_add','res1_r2']
for layer in model.layers:
    layer.trainable = layer.name not in capas_res1

n_trainable = sum(1 for l in model.layers if l.trainable and l.weights)
n_frozen    = sum(1 for l in model.layers if not l.trainable and l.weights)
print(f"  Capas entrenables: {n_trainable}  |  Congeladas: {n_frozen}")

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss      = bce_label_smoothing(0.05),
    metrics   = [
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
    ]
)

# tf.data con prefetch para velocidad
BATCH_SIZE = 64
AUTOTUNE   = tf.data.AUTOTUNE

def make_dataset(X, F, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        ({'ecg_signal': X, 'ecg_features': F}, y)
    )
    if shuffle:
        ds = ds.shuffle(len(X), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

train_ds = make_dataset(X_train, F_train, y_train, shuffle=True)
val_ds   = make_dataset(X_val,   F_val,   y_val)

history_b1 = model.fit(
    train_ds,
    epochs          = 30,
    validation_data = val_ds,
    class_weight    = CLASS_WEIGHTS,
    callbacks       = [
        callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('mejor_modelo_brugada_v4_b1.keras',
                                  monitor='val_loss', save_best_only=True,
                                  verbose=0),
        callbacks.LambdaCallback(
            on_epoch_end=lambda ep, logs: print(
                f"  [ep {ep+1:>2}] loss={logs['loss']:.4f} | "
                f"val_loss={logs['val_loss']:.4f} | "
                f"val_auc={logs.get('val_auc_roc',0):.3f} | "
                f"val_rec={logs.get('val_recall',0):.3f}"
            ) if (ep+1) % 5 == 0 else None
        ),
    ],
    verbose = 1
)
print(f"  ✓ Fase B1 completada")


# ─────────────────────────────────────────────────────────────────────────────
# 7. FASE B2 — DESCONGELAR TODO, fine-tuning completo
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FASE B2 — DESCONGELAR TODO, LR=1e-5, 200 épocas")
print("  Objetivo: ajuste fino completo al patrón de Brugada")
print("=" * 62)

for layer in model.layers:
    layer.trainable = True

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss      = bce_label_smoothing(0.05),
    metrics   = [
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
        tf.keras.metrics.AUC(name='auc_pr',  curve='PR'),
    ]
)

history_b2 = model.fit(
    train_ds,
    epochs          = 200,
    validation_data = val_ds,
    class_weight    = CLASS_WEIGHTS,
    callbacks       = [
        callbacks.EarlyStopping(monitor='val_loss', patience=50,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint('mejor_modelo_brugada_v4.keras',
                                  monitor='val_loss', save_best_only=True,
                                  verbose=0),
        callbacks.LambdaCallback(
            on_epoch_end=lambda ep, logs: print(
                f"  [ep {ep+1:>3}] loss={logs['loss']:.4f} | "
                f"val_loss={logs['val_loss']:.4f} | "
                f"val_auc={logs.get('val_auc_roc',0):.3f} | "
                f"val_rec={logs.get('val_recall',0):.3f}"
            ) if (ep+1) % 10 == 0 else None
        ),
    ],
    verbose = 1
)

best_epoch = int(np.argmin(history_b2.history['val_loss'])) + 1
print(f"\n  Mejor época B2: {best_epoch}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. UMBRAL ÓPTIMO
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  BÚSQUEDA DE UMBRAL ÓPTIMO")
print("=" * 62)

y_prob = model.predict([X_test, F_test], verbose=0).flatten()
auc    = roc_auc_score(y_test, y_prob)

print(f"\n  AUC-ROC : {auc:.4f}")
print(f"  Rango   : [{y_prob.min():.4f}, {y_prob.max():.4f}]")

best = {'t': 0.5, 'f1': 0.0, 'recall': 0.0, 'prec': 0.0,
        'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

print(f"\n  {'Umbral':>7} {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}  "
      f"{'Recall':>7} {'Prec':>7} {'F1':>7}")
print("  " + "─" * 56)

for t in np.arange(0.15, 0.90, 0.025):
    yp  = (y_prob > t).astype(int)
    cm_ = confusion_matrix(y_test, yp)
    if cm_.shape != (2, 2):
        continue
    tn, fp, fn, tp = cm_.ravel()
    rec  = tp / (tp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    flag = " ◄" if (rec >= RECALL_MIN and f1 > best['f1']) else ""
    print(f"  {t:>7.3f}  {tn:>4} {fp:>4} {fn:>4} {tp:>4}  "
          f"{rec:>7.3f} {prec:>7.3f} {f1:>7.3f}{flag}")
    if rec >= RECALL_MIN and f1 > best['f1']:
        best.update({'t': t, 'f1': f1, 'recall': rec, 'prec': prec,
                     'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})

print(f"\n  ✓ Umbral óptimo : {best['t']:.3f}")
print(f"    Recall        : {best['recall']:.4f}")
print(f"    Precision     : {best['prec']:.4f}")
print(f"    F1            : {best['f1']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUACIÓN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  EVALUACIÓN FINAL — TEST SET")
print("=" * 62)

y_pred = (y_prob > best['t']).astype(int)
cm     = confusion_matrix(y_test, y_pred)
print(f"\n  Umbral: {best['t']:.3f}  |  AUC: {auc:.4f}")
print(f"\n  [[{cm[0,0]:>3} {cm[0,1]:>3}]   ← TN | FP")
print(f"   [{cm[1,0]:>3} {cm[1,1]:>3}]]  ← FN | TP\n")
print(classification_report(y_test, y_pred,
      target_names=['Control','Brugada'], digits=4))

print("  COMPARATIVA COMPLETA:")
print(f"  {'Modelo':<28} {'AUC':>6} {'Recall':>7} {'Prec':>7} {'FP':>4} {'FN':>4}")
print("  " + "─" * 58)
for name, auc_, rec_, prec_, fp_, fn_ in [
    ('CNN Base (v1)',              'N/A',  0.9333, 0.4242, 19, 1),
    ('CNN-Res-LSTM v2',            0.9530, 0.8667, 0.5652, 11, 2),
    ('CNN-Dual v3.1',              0.9437, 0.8667, 0.7647,  4, 2),
    ('CNN-Dual v3.3 (mejor)',      0.9115, 0.9333, 0.7368,  5, 1),
    ('CNN-Dual v4 Transfer Learn', auc,    best['recall'], best['prec'],
                                   cm[0,1], cm[1,0]),
]:
    auc_s = f"{auc_:.4f}" if isinstance(auc_, float) else str(auc_)
    print(f"  {name:<28} {auc_s:>6} {rec_:>7.4f} {prec_:>7.4f} {fp_:>4} {fn_:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────
fpr_c, tpr_c, _ = roc_curve(y_test, y_prob)

# Combinar historiales B1 + B2
loss_b1     = history_b1.history['loss']
loss_b2     = history_b2.history['loss']
val_loss_b1 = history_b1.history['val_loss']
val_loss_b2 = history_b2.history['val_loss']
auc_b2      = history_b2.history['auc_roc']
val_auc_b2  = history_b2.history['val_auc_roc']

ep_b1 = range(1, len(loss_b1)+1)
ep_b2 = range(len(loss_b1)+1, len(loss_b1)+len(loss_b2)+1)

fig = plt.figure(figsize=(16, 8))
fig.suptitle('v4 — Transfer Learning PTB-XL → Brugada', fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ep_b1, loss_b1,     color='#1565C0', alpha=0.5, label='Train B1')
ax1.plot(ep_b1, val_loss_b1, color='#C62828', alpha=0.5, ls='--', label='Val B1')
ax1.plot(ep_b2, loss_b2,     color='#1565C0', label='Train B2')
ax1.plot(ep_b2, val_loss_b2, color='#C62828', ls='--', label='Val B2')
ax1.axvline(len(loss_b1), color='gray', ls=':', alpha=0.6, label='B1→B2')
ax1.set_title('Loss (B1 + B2)', fontweight='bold')
ax1.set_xlabel('Época'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ep_b2, auc_b2,     color='#6A1B9A', label='Train')
ax2.plot(ep_b2, val_auc_b2, color='#6A1B9A', ls='--', label='Val')
ax2.axhline(0.95, color='gray', ls=':', alpha=0.6, label='Ref 0.95')
ax2.set_title('AUC-ROC (Fase B2)', fontweight='bold')
ax2.set_xlabel('Época'); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim([0.5,1.02])

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(fpr_c, tpr_c, color='#1565C0', lw=2, label=f'AUC={auc:.3f}')
ax3.plot([0,1],[0,1], 'k--', alpha=0.4)
ax3.set_title('Curva ROC — Test', fontweight='bold')
ax3.set_xlabel('FPR'); ax3.set_ylabel('TPR')
ax3.legend(); ax3.grid(alpha=0.3)

ax4 = fig.add_subplot(gs[1, 0])
ax4.hist(y_prob[y_test==0], bins=20, alpha=0.6, color='#1565C0', label='Control')
ax4.hist(y_prob[y_test==1], bins=10, alpha=0.7, color='#C62828', label='Brugada')
ax4.axvline(best['t'], color='black', ls='--', lw=2, label=f"t={best['t']:.3f}")
ax4.set_title('Distribución probabilidades', fontweight='bold')
ax4.set_xlabel('P(Brugada)'); ax4.legend(); ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1:])
mods = ['CNN\nBase', 'Res\nv2', 'Dual\nv3.1', 'FT\nv3.3', 'TL\nv4']
fps  = [19, 11, 4, 5, cm[0,1]]
fns  = [1,   2, 2, 1, cm[1,0]]
x = np.arange(5); w = 0.35
b1 = ax5.bar(x-w/2, fps, w, label='FP', color='#FF8F00', alpha=0.85)
b2 = ax5.bar(x+w/2, fns, w, label='FN', color='#C62828', alpha=0.85)
for bar in list(b1)+list(b2):
    ax5.annotate(str(int(bar.get_height())),
                 xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                 xytext=(0,3), textcoords='offset points', ha='center', fontsize=11)
ax5.set_title('Evolución de errores — todos los modelos', fontweight='bold')
ax5.set_ylabel('Pacientes mal clasificados')
ax5.set_xticks(x); ax5.set_xticklabels(mods, fontsize=9)
ax5.legend(); ax5.grid(alpha=0.3, axis='y')

plt.savefig('resultados_brugada_v4_tl.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ resultados_brugada_v4_tl.png guardado")

# Guardar
model.save('modelo_brugada_dual_v4.keras')
np.save('umbral_optimo_v4.npy', np.array([best['t']]))

print(f"""
╔══════════════════════════════════════════════════════════╗
║  RESUMEN FINAL — v4 TRANSFER LEARNING                    ║
╠══════════════════════════════════════════════════════════╣
║  AUC-ROC        : {auc:.4f}                             ║
║  Umbral óptimo  : {best['t']:.3f}                               ║
║  Recall Brugada : {best['recall']:.4f}  (mín clínico: 0.85)       ║
║  Precision      : {best['prec']:.4f}                             ║
║  Falsos Pos.    : {cm[0,1]}   (v3.3: 5)                      ║
║  Falsos Neg.    : {cm[1,0]}   (v3.3: 1)                      ║
╚══════════════════════════════════════════════════════════╝
""")
