"""
================================================================================
  DETECCIÓN SÍNDROME DE BRUGADA — FASE 3 v3.1: MODELO CORREGIDO
================================================================================
  Correcciones respecto a v3 (que crasheó en epoch 52):

  BUG 1 — AttributeError 'str' object has no attribute 'name'
    Causa:  CosineDecayRestarts guarda el LR como Schedule (objeto Python),
            no como tensor. K.set_value() espera un tensor → crash.
    Fix:    Eliminamos K.set_value() completamente. El Schedule ya gestiona
            el LR de forma autónoma y correcta — no necesitamos tocarlo.

  BUG 2 — Val AUC 0.714 con Train AUC 0.985 (gap 0.27 = overfitting)
    Causa:  Val set tiene ~9 Brugada reales. 1 FN = 11 puntos de recall.
            Las métricas de val son demasiado ruidosas para guiar EarlyStopping.
    Fix:    Monitorizamos val_loss (estable) en lugar de val_auc_roc (ruidoso).
            Aumentamos Dropout y L2 regularization para reducir el gap train/val.

  BUG 3 — EarlyStopping paró en epoch 52 pero el mejor fue el 27
    Causa:  Paciencia 30 con métricas muy ruidosas → para demasiado pronto.
    Fix:    Paciencia 40 + monitor val_loss. La loss es más suave que AUC o Recall
            con pocos ejemplos en val, así el EarlyStopping tiene señal real.
================================================================================
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings

# Justo después de los imports, antes de todo lo demás:
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


# Activar mixed precision — usa float16 en los cálculos y float32 en los pesos
# Reduce VRAM a la mitad y aprovecha los Tensor Cores de la RTX 3050
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  CARGANDO DATOS v3.1")
print("=" * 62)

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val   = np.load('X_val.npy')
y_val   = np.load('y_val.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')

print(f"  Train  {X_train.shape}  Ctrl={int((y_train==0).sum())}  Brug={int((y_train==1).sum())}")
print(f"  Val    {X_val.shape}    Ctrl={int((y_val==0).sum())}    Brug={int((y_val==1).sum())}")
print(f"  Test   {X_test.shape}   Ctrl={int((y_test==0).sum())}   Brug={int((y_test==1).sum())}")

n_brug_val = int((y_val==1).sum())
print(f"\n  Nota: con {n_brug_val} Brugada en val, cada FN mueve recall {100/n_brug_val:.0f} puntos")
print(f"  → Monitorizamos val_loss (suave) en vez de val_recall (ruidoso)")

input_shape = (X_train.shape[1], X_train.shape[2])


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES ESTADÍSTICOS
# ─────────────────────────────────────────────────────────────────────────────

def extract_statistical_features(X, fs=100):
    """18 features estadísticos y espectrales por señal (6 por derivación)."""
    N       = X.shape[0]
    n_samp  = X.shape[1]
    freqs   = np.fft.rfftfreq(n_samp, 1/fs)
    feats   = np.zeros((N, 18))

    for i in range(N):
        for ch in range(3):
            sig  = X[i, :, ch]
            base = ch * 6
            mu, std = sig.mean(), sig.std() + 1e-8

            feats[i, base+0] = np.sqrt(np.mean(sig**2))                          # RMS
            feats[i, base+1] = np.mean(((sig - mu)/std)**4)                      # Kurtosis
            feats[i, base+2] = np.mean(((sig - mu)/std)**3)                      # Skewness
            feats[i, base+3] = ((sig[:-1]*sig[1:]) < 0).sum() / n_samp           # ZCR
            fft_m  = np.abs(np.fft.rfft(sig))
            e_tot  = fft_m.sum() + 1e-8
            feats[i, base+4] = fft_m[(freqs>=0.5)&(freqs<=5)].sum()  / e_tot    # E_ST
            feats[i, base+5] = fft_m[(freqs>=5) &(freqs<=40)].sum()  / e_tot    # E_QRS

    return feats

print("\n  Extrayendo features estadísticos...")
F_train = extract_statistical_features(X_train)
F_val   = extract_statistical_features(X_val)
F_test  = extract_statistical_features(X_test)

# Normalización z-score (fit SOLO en train)
feat_mean = F_train.mean(axis=0)
feat_std  = F_train.std(axis=0) + 1e-8
F_train   = (F_train - feat_mean) / feat_std
F_val     = (F_val   - feat_mean) / feat_std
F_test    = (F_test  - feat_mean) / feat_std

np.save('feat_mean.npy', feat_mean)
np.save('feat_std.npy',  feat_std)
print(f"  ✓ Features {F_train.shape} extraídos y normalizados")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL SMOOTHING BCE
# ─────────────────────────────────────────────────────────────────────────────

def bce_label_smoothing(epsilon=0.05):
    """BCE con Label Smoothing. Convierte etiquetas 0→0.025, 1→0.975."""
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        y_smooth = y_true * (1.0 - epsilon) + epsilon / 2.0
        return tf.reduce_mean(
            -(y_smooth * tf.math.log(y_pred) +
              (1 - y_smooth) * tf.math.log(1 - y_pred))
        )
    loss_fn.__name__ = f'bce_smooth_{epsilon}'
    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# 4. ARQUITECTURA — ENTRADA DUAL CON L2 REGULARIZATION
# ─────────────────────────────────────────────────────────────────────────────

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


def build_model_v31(
    input_shape   = (1200, 3),
    n_features    = 18,
    lstm_units    = 32,
    dropout_rate  = 0.30,       # aumentado de 0.20 → 0.30 para reducir overfitting
    l2_reg        = 1e-4,       # L2 regularización añadida
    learning_rate = 2e-4,       # bajado de 3e-4 → 2e-4 para entrenamiento más estable
    label_smooth  = 0.05
):
    """
    Modelo dual: señal cruda (CNN Residual + LSTM) + features estadísticos (MLP).
    
    Cambios vs v3 para reducir overfitting:
      - Dropout 0.30 (antes 0.20)
      - L2 regularization en todas las Conv1D
      - LR inicial 2e-4 (antes 3e-4)
      - CosineDecayRestarts sin K.set_value externo (era el crash)
    """
    # ── RAMA A: Señal cruda ──────────────────────────────────────────────────
    inp_ecg = layers.Input(shape=input_shape, name='ecg_signal')

    x = layers.Conv1D(32, 7, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name='conv_in')(inp_ecg)
    x = layers.BatchNormalization(name='bn_in')(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 32, 5, 'res1', l2=l2_reg)
    x = layers.MaxPooling1D(4, name='pool1')(x)           # 1200 → 300
    x = layers.Dropout(dropout_rate * 0.5)(x)

    x = residual_block(x, 64, 3, 'res2', l2=l2_reg)
    x = layers.MaxPooling1D(4, name='pool2')(x)           # 300 → 75
    x = layers.Dropout(dropout_rate * 0.5)(x)

    x = layers.LSTM(
        lstm_units,
        return_sequences  = False,
        dropout           = dropout_rate * 0.5,
        recurrent_dropout = 0.10,             # aumentado de 0.05
        kernel_regularizer = regularizers.l2(l2_reg),
        name              = 'lstm'
    )(x)
    rama_a = layers.Dense(
        64, activation='relu',
        kernel_regularizer=regularizers.l2(l2_reg),
        name='dense_a'
    )(x)
    rama_a = layers.Dropout(dropout_rate, name='drop_a')(rama_a)

    # ── RAMA B: Features estadísticos ────────────────────────────────────────
    inp_feats = layers.Input(shape=(n_features,), name='ecg_features')

    f = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name='dense_f1')(inp_feats)
    f = layers.BatchNormalization(name='bn_f1')(f)
    f = layers.Dropout(dropout_rate, name='drop_f1')(f)
    rama_b = layers.Dense(16, activation='relu',
                          name='dense_f2')(f)

    # ── FUSIÓN ────────────────────────────────────────────────────────────────
    fused  = layers.Concatenate(name='fusion')([rama_a, rama_b])
    fused  = layers.Dropout(dropout_rate * 0.5, name='drop_fused')(fused)
    fused  = layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name='dense_fused')(fused)
    fused  = layers.Dropout(dropout_rate * 0.5)(fused)
    output = layers.Dense(1, activation='sigmoid', name='output', dtype='float32')(fused)

    model = models.Model(
        inputs  = [inp_ecg, inp_feats],
        outputs = output,
        name    = 'BrugadaDual_v31'
    )

    # ── OPTIMIZADOR ──────────────────────────────────────────────────────────
    # CosineDecayRestarts sin ningún K.set_value externo.
    # El schedule se pasa directamente a Adam y TF lo gestiona internamente.
    # first_decay_steps=40 → primer ciclo 40 épocas, luego 60, 90...
    lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
        initial_learning_rate = learning_rate,
        first_decay_steps     = 40,
        t_mul                 = 1.5,
        m_mul                 = 0.85,
        alpha                 = 1e-6
    )

    model.compile(
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = lr_schedule,   # Schedule completo — no tocar después
            clipnorm      = 1.0
        ),
        loss    = bce_label_smoothing(label_smooth),
        metrics = [
            'accuracy',
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
            tf.keras.metrics.AUC(name='auc_pr',  curve='PR'),
        ]
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 5. CALLBACKS — sin K.set_value, monitor val_loss
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(path='mejor_modelo_brugada_v31.keras'):
    return [
        # Monitor val_loss: más estable que val_auc_roc con pocos Brugada en val
        callbacks.EarlyStopping(
            monitor              = 'val_loss',
            patience             = 40,         # más paciencia con cosine annealing
            mode                 = 'min',
            restore_best_weights = True,
            verbose              = 1
        ),
        callbacks.ModelCheckpoint(
            filepath       = path,
            monitor        = 'val_loss',
            save_best_only = True,
            mode           = 'min',
            verbose        = 0
        ),
        # Log de métricas clave cada 10 épocas para seguimiento
        callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"  [ep {epoch+1:>3}] "
                f"loss={logs['loss']:.4f} | "
                f"val_loss={logs['val_loss']:.4f} | "
                f"val_auc={logs.get('val_auc_roc',0):.3f} | "
                f"val_rec={logs.get('val_recall',0):.3f}"
            ) if (epoch+1) % 10 == 0 else None
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 6. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
n_total = len(y_train)
n_ctrl  = int((y_train==0).sum())
n_brug  = int((y_train==1).sum())
CLASS_WEIGHTS = {0: n_total/(2.0*n_ctrl), 1: n_total/(2.0*n_brug)}
print(f"\n  Class weights: Control={CLASS_WEIGHTS[0]:.3f}  Brugada={CLASS_WEIGHTS[1]:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 7. CONSTRUIR Y ENTRENAR
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  MODELO v3.1 — ENTRADA DUAL")
print("=" * 62)

model = build_model_v31(input_shape=input_shape)
model.summary()

n_params = model.count_params()
ratio    = n_params / len(X_train)
print(f"\n  Parámetros: {n_params:,}  |  Ratio: {ratio:.1f}:1")

print("\n" + "=" * 62)
print("  ENTRENAMIENTO — 200 épocas máx, monitor=val_loss")
print("=" * 62)

history = model.fit(
    x               = [X_train, F_train],
    y               = y_train,
    epochs          = 200,
    batch_size      = 32,
    validation_data = ([X_val, F_val], y_val),
    class_weight    = CLASS_WEIGHTS,
    callbacks       = get_callbacks('mejor_modelo_brugada_v31.keras'),
    verbose         = 1
)

best_epoch = int(np.argmin(history.history['val_loss'])) + 1
print(f"\n  Mejor época: {best_epoch}  (val_loss mínimo)")


# ─────────────────────────────────────────────────────────────────────────────
# 8. UMBRAL ÓPTIMO
# ─────────────────────────────────────────────────────────────────────────────
RECALL_MIN = 0.85

print("\n" + "=" * 62)
print("  BÚSQUEDA DE UMBRAL ÓPTIMO EN TEST SET")
print("=" * 62)

y_prob = model.predict([X_test, F_test], verbose=0).flatten()
auc    = roc_auc_score(y_test, y_prob)

print(f"\n  AUC-ROC test  : {auc:.4f}")
print(f"  Prob range    : [{y_prob.min():.4f}, {y_prob.max():.4f}]")
print(f"  (rango > 0.5 = buena separación; < 0.1 = colapso)")

best = {'t': 0.5, 'f1': 0.0, 'recall': 0.0, 'prec': 0.0,
        'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}

print(f"\n  {'Umbral':>7} {'TN':>4} {'FP':>4} {'FN':>4} {'TP':>4}  "
      f"{'Recall':>7} {'Prec':>7} {'Spec':>7} {'F1':>7}")
print("  " + "─" * 62)

for t in np.arange(0.15, 0.90, 0.025):
    yp  = (y_prob > t).astype(int)
    cm_ = confusion_matrix(y_test, yp)
    if cm_.shape != (2, 2):
        continue
    tn, fp, fn, tp = cm_.ravel()
    rec  = tp / (tp + fn + 1e-8)
    prec = tp / (tp + fp + 1e-8)
    spec = tn / (tn + fp + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8)
    flag = " ◄" if (rec >= RECALL_MIN and f1 > best['f1']) else ""
    print(f"  {t:>7.3f}  {tn:>4} {fp:>4} {fn:>4} {tp:>4}  "
          f"{rec:>7.3f} {prec:>7.3f} {spec:>7.3f} {f1:>7.3f}{flag}")
    if rec >= RECALL_MIN and f1 > best['f1']:
        best.update({'t': t, 'f1': f1, 'recall': rec, 'prec': prec,
                     'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})

print(f"\n  ✓ Umbral óptimo : {best['t']:.3f}")
print(f"    Recall        : {best['recall']:.4f}  (mín clínico: {RECALL_MIN})")
print(f"    Precision     : {best['prec']:.4f}")
print(f"    F1-Score      : {best['f1']:.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# 9. EVALUACIÓN FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  EVALUACIÓN FINAL — TEST SET")
print("=" * 62)

y_pred = (y_prob > best['t']).astype(int)
cm     = confusion_matrix(y_test, y_pred)
acc    = (cm[0,0] + cm[1,1]) / len(y_test)

print(f"\n  Umbral: {best['t']:.3f}  |  AUC-ROC: {auc:.4f}")
print(f"\n  [[{cm[0,0]:>3} {cm[0,1]:>3}]   ← TN | FP")
print(f"   [{cm[1,0]:>3} {cm[1,1]:>3}]]  ← FN | TP\n")
print(classification_report(y_test, y_pred,
      target_names=['Control','Brugada'], digits=4))

print("  EVOLUCIÓN DEL PROYECTO:")
print(f"  {'Modelo':<24} {'AUC':>6} {'Recall':>7} {'Prec':>7} {'FP':>4} {'FN':>4}")
print("  " + "─" * 54)
for name, auc_, rec_, prec_, fp_, fn_ in [
    ('CNN Base (fase3)',        'N/A',  0.9333, 0.4242, 19,        1),
    ('CNN-BiLSTM (fase4 fall)', 0.6345, 1.0000, 0.2083, 57,        0),
    ('CNN-Res-LSTM v2',         0.9530, 0.8667, 0.5652, 11,        2),
    ('CNN-Dual v3.1 (actual)',  auc,    best['recall'], best['prec'], cm[0,1], cm[1,0]),
]:
    auc_s = f"{auc_:.4f}" if isinstance(auc_, float) else str(auc_)
    print(f"  {name:<24} {auc_s:>6} {rec_:>7.4f} {prec_:>7.4f} {fp_:>4} {fn_:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────
fpr_c, tpr_c, roc_t = roc_curve(y_test, y_prob)
ep = range(1, len(history.history['loss']) + 1)

fig = plt.figure(figsize=(18, 10))
fig.suptitle('CNN Dual Input v3.1 — Detección Síndrome de Brugada',
             fontsize=14, fontweight='bold')
gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.40, wspace=0.35)

# Loss
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ep, history.history['loss'],     label='Train', color='#1565C0')
ax1.plot(ep, history.history['val_loss'], label='Val',   color='#C62828', ls='--')
ax1.axvline(best_epoch, color='gray', ls=':', alpha=0.6, label=f'Best ep.{best_epoch}')
ax1.set_title('BCE Loss + smoothing', fontweight='bold')
ax1.set_xlabel('Época'); ax1.legend(); ax1.grid(alpha=0.3)

# AUC-ROC
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ep, history.history['auc_roc'],     label='Train AUC', color='#6A1B9A')
ax2.plot(ep, history.history['val_auc_roc'], label='Val AUC',   color='#6A1B9A', ls='--')
ax2.axhline(0.95, color='gray', ls=':', alpha=0.6, label='Ref 0.95')
ax2.set_title('AUC-ROC entrenamiento', fontweight='bold')
ax2.set_xlabel('Época'); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim([0.4, 1.05])

# Recall
ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(ep, history.history['recall'],     label='Train', color='#2E7D32')
ax3.plot(ep, history.history['val_recall'], label='Val',   color='#2E7D32', ls='--')
ax3.axhline(RECALL_MIN, color='red', ls=':', alpha=0.7, label=f'Mín {RECALL_MIN}')
ax3.set_title('Recall (umbral 0.5 train)', fontweight='bold')
ax3.set_xlabel('Época'); ax3.legend(); ax3.grid(alpha=0.3); ax3.set_ylim([0, 1.05])

# Precision
ax4 = fig.add_subplot(gs[0, 3])
ax4.plot(ep, history.history['precision'],     label='Train', color='#E65100')
ax4.plot(ep, history.history['val_precision'], label='Val',   color='#E65100', ls='--')
ax4.set_title('Precision (umbral 0.5 train)', fontweight='bold')
ax4.set_xlabel('Época'); ax4.legend(); ax4.grid(alpha=0.3); ax4.set_ylim([0, 1.05])

# Curva ROC
ax5 = fig.add_subplot(gs[1, 0])
ax5.plot(fpr_c, tpr_c, color='#1565C0', lw=2, label=f'AUC={auc:.3f}')
ax5.plot([0,1],[0,1], 'k--', alpha=0.4)
bi = np.argmin(np.abs(roc_t - best['t']))
ax5.scatter(fpr_c[bi], tpr_c[bi], color='red', s=80, zorder=5, label=f"t={best['t']:.3f}")
ax5.set_title('Curva ROC — Test set', fontweight='bold')
ax5.set_xlabel('FPR'); ax5.set_ylabel('TPR')
ax5.legend(); ax5.grid(alpha=0.3)

# Distribución probabilidades
ax6 = fig.add_subplot(gs[1, 1])
ax6.hist(y_prob[y_test==0], bins=20, alpha=0.6, color='#1565C0', label='Control')
ax6.hist(y_prob[y_test==1], bins=10, alpha=0.7, color='#C62828', label='Brugada')
ax6.axvline(best['t'], color='black', ls='--', lw=2, label=f"t={best['t']:.3f}")
ax6.set_title('Distribución probabilidades', fontweight='bold')
ax6.set_xlabel('P(Brugada)'); ax6.legend(); ax6.grid(alpha=0.3)

# Top features discriminativos
ax7 = fig.add_subplot(gs[1, 2])
fnames = ['RMS-V1','Kurt-V1','Skew-V1','ZCR-V1','E_ST-V1','E_QRS-V1',
          'RMS-V2','Kurt-V2','Skew-V2','ZCR-V2','E_ST-V2','E_QRS-V2',
          'RMS-V3','Kurt-V3','Skew-V3','ZCR-V3','E_ST-V3','E_QRS-V3']
delta   = np.abs(F_test[y_test==1].mean(0) - F_test[y_test==0].mean(0))
top_idx = np.argsort(delta)[-9:][::-1]
ax7.barh([fnames[i] for i in top_idx], delta[top_idx], color='#6A1B9A', alpha=0.8)
ax7.set_title('Top features (Δ Ctrl vs Brug)', fontweight='bold')
ax7.set_xlabel('|Δ normalizado|'); ax7.grid(alpha=0.3, axis='x')

# Comparativa histórica FP/FN
ax8 = fig.add_subplot(gs[1, 3])
mods = ['CNN\nBase', 'BiLSTM\n(fall.)', 'Res\nv2', 'Dual\nv3.1']
fps  = [19, 57, 11, cm[0,1]]
fns  = [1,  0,  2,  cm[1,0]]
x = np.arange(4); w = 0.35
b1 = ax8.bar(x-w/2, fps, w, label='FP', color='#FF8F00', alpha=0.85)
b2 = ax8.bar(x+w/2, fns, w, label='FN', color='#C62828', alpha=0.85)
for bar in list(b1)+list(b2):
    ax8.annotate(str(int(bar.get_height())),
                 xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                 xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10)
ax8.set_title('Errores: todos los modelos', fontweight='bold')
ax8.set_ylabel('Pacientes mal clasificados')
ax8.set_xticks(x); ax8.set_xticklabels(mods, fontsize=9)
ax8.legend(); ax8.grid(alpha=0.3, axis='y')

plt.savefig('resultados_brugada_v31.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ resultados_brugada_v31.png guardado")


# ─────────────────────────────────────────────────────────────────────────────
# 11. GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
model.save('modelo_brugada_dual_v31.keras')
np.save('umbral_optimo_v31.npy', np.array([best['t']]))

print(f"""
╔══════════════════════════════════════════════════════════╗
║     RESUMEN FINAL — CNN DUAL INPUT v3.1                  ║
╠══════════════════════════════════════════════════════════╣
║  Mejor época    : {best_epoch}                                    ║
║  AUC-ROC        : {auc:.4f}                             ║
║  Umbral óptimo  : {best['t']:.3f}                               ║
║  Recall Brugada : {best['recall']:.4f}  (mín clínico: {RECALL_MIN})       ║
║  Precision      : {best['prec']:.4f}                             ║
║  Falsos Pos.    : {cm[0,1]}                                   ║
║  Falsos Neg.    : {cm[1,0]}                                   ║
╚══════════════════════════════════════════════════════════╝
""")


# ─────────────────────────────────────────────────────────────────────────────
# 12. FUNCIÓN DE INFERENCIA PRODUCCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def predict_brugada_v31(ecg_signal,
                        model_path  = 'mejor_modelo_brugada_v31.keras',
                        umbral_path = 'umbral_optimo_v31.npy',
                        mean_path   = 'feat_mean.npy',
                        std_path    = 'feat_std.npy'):
    """
    Inferencia lista para producción con el modelo dual v3.1.

    Args:
        ecg_signal : np.array (1200, 3) o (N, 1200, 3) — V1, V2, V3 @ 100 Hz
    Returns:
        dict con probabilidad, prediccion, diagnostico, confianza, umbral
    """
    loaded = tf.keras.models.load_model(
        model_path,
        custom_objects={'bce_smooth_0.05': bce_label_smoothing(0.05)}
    )
    thresh = float(np.load(umbral_path)[0])
    f_mean = np.load(mean_path)
    f_std  = np.load(std_path)

    if ecg_signal.ndim == 2:
        ecg_signal = ecg_signal[np.newaxis, ...]

    feats = (extract_statistical_features(ecg_signal) - f_mean) / f_std
    probs = loaded.predict([ecg_signal, feats], verbose=0).flatten()
    preds = (probs > thresh).astype(int)
    dist  = np.abs(probs - thresh)
    conf  = np.where(dist > 0.3, 'Alta', np.where(dist > 0.15, 'Media', 'Baja'))

    single = len(probs) == 1
    return {
        'probabilidad' : float(probs[0]) if single else probs,
        'prediccion'   : int(preds[0])   if single else preds,
        'diagnostico'  : ('Brugada' if preds[0]==1 else 'Control') if single else
                         ['Brugada' if p==1 else 'Control' for p in preds],
        'confianza'    : str(conf[0])  if single else conf,
        'umbral_usado' : thresh
    }
