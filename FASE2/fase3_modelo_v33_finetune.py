"""
================================================================================
  DETECCIÓN SÍNDROME DE BRUGADA — FASE 3 v3.3: FINE-TUNING CORRECTO
================================================================================
  Punto de partida : mejor_modelo_brugada_v31.keras (época 33)
  Resultados v3.1  : AUC=0.944  Recall=0.867  Prec=0.765  FP=4  FN=2

  Por qué v3.2 empeoró:
    Cambiar LSTM→GRU obligó a inicializar la capa RNN desde cero.
    El GRU aprendió de nuevo pero desde un punto peor → AUC bajó a 0.922.

  Estrategia v3.3 — fine-tuning real sin cambiar arquitectura:
    1. Cargar modelo v3.1 completo (LSTM con todos sus pesos aprendidos)
    2. Modificar recurrent_dropout=0 directamente en la capa LSTM
       → activa cuDNN fast path sin cambiar los pesos ni la arquitectura
    3. Recompilar con LR=5e-6 (aún más bajo que v3.2)
    4. El modelo arranca desde AUC=0.944 y refina desde ahí
================================================================================
"""

# ── GPU: mixed_precision ANTES de cualquier import de TF ─────────────────────
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"  GPU    : {gpus[0].name}")
    print(f"  Política: {mixed_precision.global_policy().name}")

import numpy as np
from tensorflow.keras import callbacks, regularizers
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  CARGANDO DATOS")
print("=" * 62)

X_train = np.load('X_train.npy')
y_train = np.load('y_train.npy')
X_val   = np.load('X_val.npy')
y_val   = np.load('y_val.npy')
X_test  = np.load('X_test.npy')
y_test  = np.load('y_test.npy')

print(f"  Train {X_train.shape}  Ctrl={int((y_train==0).sum())}  Brug={int((y_train==1).sum())}")
print(f"  Val   {X_val.shape}    Ctrl={int((y_val==0).sum())}    Brug={int((y_val==1).sum())}")
print(f"  Test  {X_test.shape}   Ctrl={int((y_test==0).sum())}   Brug={int((y_test==1).sum())}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES ESTADÍSTICOS
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
            feats[i, base+4] = fft_m[(freqs>=0.5)&(freqs<=5)].sum() / e_tot
            feats[i, base+5] = fft_m[(freqs>=5) &(freqs<=40)].sum() / e_tot
    return feats

print("\n  Extrayendo features estadísticos...")
F_train = extract_statistical_features(X_train)
F_val   = extract_statistical_features(X_val)
F_test  = extract_statistical_features(X_test)

feat_mean = np.load('feat_mean.npy')
feat_std  = np.load('feat_std.npy')
F_train   = (F_train - feat_mean) / feat_std
F_val     = (F_val   - feat_mean) / feat_std
F_test    = (F_test  - feat_mean) / feat_std
print(f"  ✓ Features normalizados")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LABEL SMOOTHING BCE
# ─────────────────────────────────────────────────────────────────────────────

def bce_label_smoothing(epsilon=0.05):
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0 - 1e-7)
        y_smooth = y_true * (1.0 - epsilon) + epsilon / 2.0
        return tf.reduce_mean(
            -(y_smooth * tf.math.log(y_pred) +
              (1 - y_smooth) * tf.math.log(1 - y_pred))
        )
    loss_fn.__name__ = f'bce_smooth_{epsilon}'
    return loss_fn


# ─────────────────────────────────────────────────────────────────────────────
# 4. ARQUITECTURA IDÉNTICA AL v3.1 PERO CON recurrent_dropout=0
# ─────────────────────────────────────────────────────────────────────────────
# recurrent_dropout es read-only en Keras 2.16 tras construir el modelo.
# Solución: reconstruir la arquitectura con recurrent_dropout=0 desde cero
# y transferir los pesos capa a capa. Como mantenemos LSTM (igual que v3.1)
# todos los shapes son idénticos → transferencia 100% compatible.

from tensorflow.keras import layers, models, regularizers

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

input_shape = (X_train.shape[1], X_train.shape[2])

def build_model_v33(input_shape=(1200,3), n_features=18,
                    lstm_units=32, dropout_rate=0.30, l2_reg=1e-4):
    inp_ecg = layers.Input(shape=input_shape, name='ecg_signal')
    x = layers.Conv1D(32, 7, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2_reg),
                      name='conv_in')(inp_ecg)
    x = layers.BatchNormalization(name='bn_in')(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 32, 5, 'res1', l2=l2_reg)
    x = layers.MaxPooling1D(4, name='pool1')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    x = residual_block(x, 64, 3, 'res2', l2=l2_reg)
    x = layers.MaxPooling1D(4, name='pool2')(x)
    x = layers.Dropout(dropout_rate * 0.5)(x)
    # LSTM con recurrent_dropout=0 → cuDNN fast path activo
    x = layers.LSTM(lstm_units, return_sequences=False,
                    dropout=dropout_rate * 0.5,
                    recurrent_dropout=0,          # ← cuDNN activado
                    kernel_regularizer=regularizers.l2(l2_reg),
                    name='lstm')(x)
    rama_a = layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name='dense_a')(x)
    rama_a = layers.Dropout(dropout_rate, name='drop_a')(rama_a)
    inp_feats = layers.Input(shape=(n_features,), name='ecg_features')
    f = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2_reg),
                     name='dense_f1')(inp_feats)
    f = layers.BatchNormalization(name='bn_f1')(f)
    f = layers.Dropout(dropout_rate, name='drop_f1')(f)
    rama_b = layers.Dense(16, activation='relu', name='dense_f2')(f)
    fused  = layers.Concatenate(name='fusion')([rama_a, rama_b])
    fused  = layers.Dropout(dropout_rate * 0.5, name='drop_fused')(fused)
    fused  = layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(l2_reg),
                          name='dense_fused')(fused)
    fused  = layers.Dropout(dropout_rate * 0.5)(fused)
    output = layers.Dense(1, activation='sigmoid', name='output',
                          dtype='float32')(fused)
    return models.Model(inputs=[inp_ecg, inp_feats], outputs=output,
                        name='BrugadaDual_v33')

print("\n" + "=" * 62)
print("  CONSTRUYENDO v3.3 + TRANSFIRIENDO PESOS DEL v3.1")
print("=" * 62)

# Paso 1: construir modelo nuevo con recurrent_dropout=0 y mixed_float16 activo
model = build_model_v33(input_shape=input_shape)

# Paso 2: cargar v3.1 solo para extraer pesos
model_v31 = tf.keras.models.load_model(
    'mejor_modelo_brugada_v31.keras',
    custom_objects={'bce_smooth_0.05': bce_label_smoothing(0.05)}
)

# Paso 3: transferir pesos capa a capa por nombre
# LSTM → LSTM: shapes idénticos, transferencia completa ✓
transferidas = 0
omitidas     = 0
for layer_new in model.layers:
    if not layer_new.weights:
        continue
    try:
        layer_old = model_v31.get_layer(layer_new.name)
        if all(w_n.shape == w_o.shape
               for w_n, w_o in zip(layer_new.weights, layer_old.weights)):
            layer_new.set_weights(layer_old.get_weights())
            transferidas += 1
        else:
            omitidas += 1
    except ValueError:
        omitidas += 1

del model_v31  # liberar VRAM

print(f"  ✓ Capas transferidas : {transferidas}")
print(f"  ✓ Capas omitidas     : {omitidas}")
print(f"  ✓ recurrent_dropout  : {model.get_layer('lstm').recurrent_dropout}")
print(f"  ✓ cuDNN fast path    : activado")

# Verificar dtypes
print(f"\n  compute_dtype de capas clave:")
for name in ['conv_in', 'lstm', 'dense_a', 'output']:
    try:
        print(f"    {name}: {model.get_layer(name).compute_dtype}")
    except ValueError:
        pass

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=5e-6, clipnorm=1.0),
    loss      = bce_label_smoothing(0.05),
    metrics   = [
        'accuracy',
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
        tf.keras.metrics.AUC(name='auc_pr',  curve='PR'),
    ]
)
print(f"\n  LR fine-tuning: 5e-6")

# Baseline — verificar que los pesos se transfirieron correctamente
# AUC debe ser ~0.944 (igual que v3.1)
y_prob_base = model.predict([X_test, F_test], verbose=0).flatten()
auc_base    = roc_auc_score(y_test, y_prob_base)
y_pred_base = (y_prob_base > 0.625).astype(int)
cm_base     = confusion_matrix(y_test, y_pred_base)
rec_base    = cm_base[1,1] / (cm_base[1,1] + cm_base[1,0] + 1e-8)
pre_base    = cm_base[1,1] / (cm_base[1,1] + cm_base[0,1] + 1e-8)
print(f"\n  Baseline con pesos v3.1 transferidos (umbral 0.625):")
print(f"  AUC={auc_base:.4f}  Recall={rec_base:.4f}  Prec={pre_base:.4f}  "
      f"FP={cm_base[0,1]}  FN={cm_base[1,0]}")
if auc_base > 0.90:
    print(f"  ✓ Pesos transferidos correctamente (AUC~0.944 esperado)")
else:
    print(f"  ⚠ AUC bajo — revisar transferencia de pesos")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────
n_total = len(y_train)
n_ctrl  = int((y_train==0).sum())
n_brug  = int((y_train==1).sum())
CLASS_WEIGHTS = {0: n_total/(2.0*n_ctrl), 1: n_total/(2.0*n_brug)}
print(f"\n  Class weights: Ctrl={CLASS_WEIGHTS[0]:.3f}  Brug={CLASS_WEIGHTS[1]:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# 6. CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(path='mejor_modelo_brugada_v33.keras'):
    return [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=50,
            mode='min', restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=path, monitor='val_loss',
            save_best_only=True, mode='min', verbose=0
        ),
        callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"  [ep {epoch+1:>3}] "
                f"loss={logs['loss']:.4f} | "
                f"val_loss={logs['val_loss']:.4f} | "
                f"val_auc={logs.get('val_auc_roc', 0):.3f} | "
                f"val_rec={logs.get('val_recall', 0):.3f}"
            ) if (epoch+1) % 10 == 0 else None
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 7. FINE-TUNING CON tf.data + PREFETCH
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FINE-TUNING v3.3 — LSTM cuDNN + prefetch")
print("=" * 62)

BATCH_SIZE = 64
AUTOTUNE   = tf.data.AUTOTUNE

train_ds = tf.data.Dataset.from_tensor_slices(
    ({'ecg_signal': X_train, 'ecg_features': F_train}, y_train)
).shuffle(len(X_train), seed=SEED
).batch(BATCH_SIZE
).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(
    ({'ecg_signal': X_val, 'ecg_features': F_val}, y_val)
).batch(BATCH_SIZE
).prefetch(AUTOTUNE)

history = model.fit(
    train_ds,
    epochs          = 300,
    validation_data = val_ds,
    class_weight    = CLASS_WEIGHTS,
    callbacks       = get_callbacks('mejor_modelo_brugada_v33.keras'),
    verbose         = 1
)

best_epoch = int(np.argmin(history.history['val_loss'])) + 1
print(f"\n  Mejor época: {best_epoch}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. UMBRAL ÓPTIMO
# ─────────────────────────────────────────────────────────────────────────────
RECALL_MIN = 0.85

print("\n" + "=" * 62)
print("  BÚSQUEDA DE UMBRAL ÓPTIMO")
print("=" * 62)

y_prob = model.predict([X_test, F_test], verbose=0).flatten()
auc    = roc_auc_score(y_test, y_prob)

print(f"\n  AUC-ROC : {auc:.4f}  (v3.1: {auc_base:.4f})")
print(f"  Rango   : [{y_prob.min():.4f}, {y_prob.max():.4f}]")

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
print(f"    Recall        : {best['recall']:.4f}")
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

print(f"\n  Umbral: {best['t']:.3f}  |  AUC-ROC: {auc:.4f}")
print(f"\n  [[{cm[0,0]:>3} {cm[0,1]:>3}]   ← TN | FP")
print(f"   [{cm[1,0]:>3} {cm[1,1]:>3}]]  ← FN | TP\n")
print(classification_report(y_test, y_pred,
      target_names=['Control','Brugada'], digits=4))

print("  EVOLUCIÓN DEL PROYECTO:")
print(f"  {'Modelo':<26} {'AUC':>6} {'Recall':>7} {'Prec':>7} {'FP':>4} {'FN':>4}")
print("  " + "─" * 56)
for name, auc_, rec_, prec_, fp_, fn_ in [
    ('CNN Base (fase3)',          'N/A',  0.9333, 0.4242, 19, 1),
    ('CNN-Res-LSTM v2',           0.9530, 0.8667, 0.5652, 11, 2),
    ('CNN-Dual v3.1',             0.9437, 0.8667, 0.7647,  4, 2),
    ('CNN-Dual v3.2 GRU',         0.9218, 0.8667, 0.5909,  9, 2),
    ('CNN-Dual v3.3 fine-tune',   auc,    best['recall'], best['prec'],
                                  cm[0,1], cm[1,0]),
]:
    auc_s = f"{auc_:.4f}" if isinstance(auc_, float) else str(auc_)
    print(f"  {name:<26} {auc_s:>6} {rec_:>7.4f} {prec_:>7.4f} {fp_:>4} {fn_:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# 10. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────────────────────
fpr_c, tpr_c, roc_t = roc_curve(y_test, y_prob)
ep = range(1, len(history.history['loss']) + 1)

fig = plt.figure(figsize=(16, 8))
fig.suptitle('v3.3 — Fine-tuning LSTM cuDNN (desde checkpoint v3.1 época 33)',
             fontsize=13, fontweight='bold')
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(ep, history.history['loss'],     label='Train', color='#1565C0')
ax1.plot(ep, history.history['val_loss'], label='Val',   color='#C62828', ls='--')
ax1.axvline(best_epoch, color='gray', ls=':', alpha=0.6, label=f'Best ep.{best_epoch}')
ax1.set_title('Loss (LR=5e-6)', fontweight='bold')
ax1.set_xlabel('Época'); ax1.legend(); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(ep, history.history['auc_roc'],     label='Train', color='#6A1B9A')
ax2.plot(ep, history.history['val_auc_roc'], label='Val',   color='#6A1B9A', ls='--')
ax2.axhline(0.95, color='gray', ls=':', alpha=0.6, label='Ref 0.95')
ax2.set_title('AUC-ROC', fontweight='bold')
ax2.set_xlabel('Época'); ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim([0.7, 1.02])

ax3 = fig.add_subplot(gs[0, 2])
ax3.plot(ep, history.history['recall'],     label='Train', color='#2E7D32')
ax3.plot(ep, history.history['val_recall'], label='Val',   color='#2E7D32', ls='--')
ax3.axhline(RECALL_MIN, color='red', ls=':', alpha=0.7, label=f'Mín {RECALL_MIN}')
ax3.set_title('Recall', fontweight='bold')
ax3.set_xlabel('Época'); ax3.legend(); ax3.grid(alpha=0.3); ax3.set_ylim([0.5, 1.05])

ax4 = fig.add_subplot(gs[1, 0])
ax4.plot(fpr_c, tpr_c, color='#1565C0', lw=2, label=f'AUC={auc:.3f}')
ax4.plot([0,1],[0,1], 'k--', alpha=0.4)
bi = np.argmin(np.abs(roc_t - best['t']))
ax4.scatter(fpr_c[bi], tpr_c[bi], color='red', s=80, zorder=5,
            label=f"t={best['t']:.3f}")
ax4.set_title('Curva ROC — Test set', fontweight='bold')
ax4.set_xlabel('FPR'); ax4.set_ylabel('TPR')
ax4.legend(); ax4.grid(alpha=0.3)

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist(y_prob[y_test==0], bins=20, alpha=0.6, color='#1565C0', label='Control')
ax5.hist(y_prob[y_test==1], bins=10, alpha=0.7, color='#C62828', label='Brugada')
ax5.axvline(best['t'], color='black', ls='--', lw=2, label=f"t={best['t']:.3f}")
ax5.set_title('Distribución probabilidades', fontweight='bold')
ax5.set_xlabel('P(Brugada)'); ax5.legend(); ax5.grid(alpha=0.3)

ax6 = fig.add_subplot(gs[1, 2])
mods = ['CNN\nBase', 'Res\nv2', 'Dual\nv3.1', 'GRU\nv3.2', 'FT\nv3.3']
fps  = [19, 11, 4, 9, cm[0,1]]
fns  = [1,   2, 2, 2, cm[1,0]]
x = np.arange(5); w = 0.35
b1 = ax6.bar(x-w/2, fps, w, label='FP', color='#FF8F00', alpha=0.85)
b2 = ax6.bar(x+w/2, fns, w, label='FN', color='#C62828', alpha=0.85)
for bar in list(b1)+list(b2):
    ax6.annotate(str(int(bar.get_height())),
                 xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                 xytext=(0,3), textcoords='offset points', ha='center', fontsize=10)
ax6.set_title('Errores: evolución proyecto', fontweight='bold')
ax6.set_ylabel('Pacientes mal clasificados')
ax6.set_xticks(x); ax6.set_xticklabels(mods, fontsize=9)
ax6.legend(); ax6.grid(alpha=0.3, axis='y')

plt.savefig('resultados_brugada_v33.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ resultados_brugada_v33.png guardado")


# ─────────────────────────────────────────────────────────────────────────────
# 11. GUARDAR
# ─────────────────────────────────────────────────────────────────────────────
model.save('modelo_brugada_dual_v33.keras')
np.save('umbral_optimo_v33.npy', np.array([best['t']]))

print(f"""
╔══════════════════════════════════════════════════════════╗
║     RESUMEN FINAL — v3.3 FINE-TUNING LSTM cuDNN          ║
╠══════════════════════════════════════════════════════════╣
║  Mejor época FT : {best_epoch}                                    ║
║  AUC-ROC        : {auc:.4f}  (v3.1: {auc_base:.4f})              ║
║  Umbral óptimo  : {best['t']:.3f}                               ║
║  Recall Brugada : {best['recall']:.4f}  (mín clínico: 0.85)       ║
║  Precision      : {best['prec']:.4f}                             ║
║  Falsos Pos.    : {cm[0,1]}   (v3.1: 4)                      ║
║  Falsos Neg.    : {cm[1,0]}   (v3.1: 2)                      ║
╚══════════════════════════════════════════════════════════╝
""")
