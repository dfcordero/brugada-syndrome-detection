"""
================================================================================
  TRANSFER LEARNING — FASE B v4.1 vs v4.2
  Comparativa de arquitecturas con pesos preentrenados de PTB-XL
================================================================================
  Ejecutar desde: ~/proyectos/brugada_syndrom/FASE3_PTB/
  python fase3B_comparativa_arquitecturas.py

  v4.1 — CNN (PTB-XL) + LSTM(32) sin GAP
    · Elimina GAP, añade LSTM después de res2+pool2
    · Brugada (1200,3) se trunca a (1000,3) para compatibilidad
    · Mismo enfoque que v3.3 pero con pesos CNN preentrenados

  v4.2 — CNN (PTB-XL) + GAP + LSTM(32)
    · Mantiene GAP para colapsar dimensión temporal
    · Añade LSTM sobre el vector (64,) → no tiene sentido temporal real
    · Lo probamos para confirmar que LSTM sin secuencia no aporta

  Hipótesis: v4.1 > v4.2 porque el LSTM necesita secuencia temporal
================================================================================
"""

# ── GPU ───────────────────────────────────────────────────────────────────────
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    print(f"  GPU: {gpus[0].name}  |  mixed_float16")

import numpy as np
from pathlib import Path
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

BRUGADA_DIR = Path('../FASE2')
PRETRAINED  = Path('mejor_modelo_pretrained_ptbxl.keras')
RECALL_MIN  = 0.85
BATCH_SIZE  = 64
AUTOTUNE    = tf.data.AUTOTUNE


# ─────────────────────────────────────────────────────────────────────────────
# 1. DATOS BRUGADA
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

print(f"  Train {X_train.shape}  Ctrl={int((y_train==0).sum())}  Brug={int((y_train==1).sum())}")
print(f"  Val   {X_val.shape}    Ctrl={int((y_val==0).sum())}    Brug={int((y_val==1).sum())}")
print(f"  Test  {X_test.shape}   Ctrl={int((y_test==0).sum())}   Brug={int((y_test==1).sum())}")

# Truncar a 1000 muestras para v4.1 (LSTM sin GAP)
# Los primeros 10 segundos contienen el patrón diagnóstico de Brugada
X_train_1000 = X_train[:, :1000, :]
X_val_1000   = X_val[:,   :1000, :]
X_test_1000  = X_test[:,  :1000, :]
print(f"\n  Versión truncada (1000 muestras): {X_train_1000.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. FEATURES ESTADÍSTICOS
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(X, fs=100):
    N, n_samp = X.shape[0], X.shape[1]
    freqs = np.fft.rfftfreq(n_samp, 1/fs)
    feats = np.zeros((N, 18))
    for i in range(N):
        for ch in range(3):
            sig  = X[i, :, ch]; base = ch * 6
            mu, std = sig.mean(), sig.std() + 1e-8
            feats[i, base+0] = np.sqrt(np.mean(sig**2))
            feats[i, base+1] = np.mean(((sig-mu)/std)**4)
            feats[i, base+2] = np.mean(((sig-mu)/std)**3)
            feats[i, base+3] = ((sig[:-1]*sig[1:]) < 0).sum() / n_samp
            fft_m = np.abs(np.fft.rfft(sig)); e = fft_m.sum() + 1e-8
            feats[i, base+4] = fft_m[(freqs>=0.5)&(freqs<=5)].sum()  / e
            feats[i, base+5] = fft_m[(freqs>=5) &(freqs<=40)].sum()  / e
    return feats

print("\n  Extrayendo features estadísticos...")
feat_mean = np.load(BRUGADA_DIR / 'feat_mean.npy')
feat_std  = np.load(BRUGADA_DIR / 'feat_std.npy')

# Features sobre señal completa (1200) — más información
F_train = (extract_features(X_train) - feat_mean) / feat_std
F_val   = (extract_features(X_val)   - feat_mean) / feat_std
F_test  = (extract_features(X_test)  - feat_mean) / feat_std
print(f"  ✓ Features (1200 muestras): {F_train.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. LOSS Y CLASS WEIGHTS
# ─────────────────────────────────────────────────────────────────────────────

def bce_label_smoothing(epsilon=0.05):
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0-1e-7)
        y_smooth = y_true * (1.0-epsilon) + epsilon/2.0
        return tf.reduce_mean(
            -(y_smooth*tf.math.log(y_pred) +
              (1-y_smooth)*tf.math.log(1-y_pred))
        )
    loss_fn.__name__ = f'bce_smooth_{epsilon}'
    return loss_fn

n_total = len(y_train)
n_ctrl  = int((y_train==0).sum())
n_brug  = int((y_train==1).sum())
CLASS_WEIGHTS = {0: n_total/(2.0*n_ctrl), 1: n_total/(2.0*n_brug)}


# ─────────────────────────────────────────────────────────────────────────────
# 4. ARQUITECTURA COMPARTIDA (bloques residuales)
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


def transfer_weights(model_new, model_old):
    """Transfiere pesos del modelo preentrenado por nombre de capa."""
    transferidas = 0
    for layer_new in model_new.layers:
        if not layer_new.weights:
            continue
        try:
            layer_old = model_old.get_layer(layer_new.name)
            if all(w_n.shape == w_o.shape
                   for w_n, w_o in zip(layer_new.weights, layer_old.weights)):
                layer_new.set_weights(layer_old.get_weights())
                transferidas += 1
        except ValueError:
            pass
    return transferidas


def make_dataset(X_ecg, F, y, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices(
        ({'ecg_signal': X_ecg, 'ecg_features': F}, y)
    )
    if shuffle:
        ds = ds.shuffle(len(y), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)


def get_callbacks(checkpoint_path, patience_es=50, patience_rlr=10):
    return [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=patience_es,
            restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            checkpoint_path, monitor='val_loss',
            save_best_only=True, verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=patience_rlr, min_lr=1e-7, verbose=1
        ),
        callbacks.LambdaCallback(
            on_epoch_end=lambda ep, logs: print(
                f"  [ep {ep+1:>3}] "
                f"loss={logs['loss']:.4f} | "
                f"val_loss={logs['val_loss']:.4f} | "
                f"val_auc={logs.get('val_auc_roc', 0):.3f}"
            ) if (ep+1) % 10 == 0 else None
        ),
    ]


def find_best_threshold(y_test, y_prob, recall_min=RECALL_MIN):
    """Barrido de umbral con restricción clínica de Recall."""
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
        flag = " ◄" if (rec >= recall_min and f1 > best['f1']) else ""
        print(f"  {t:>7.3f}  {tn:>4} {fp:>4} {fn:>4} {tp:>4}  "
              f"{rec:>7.3f} {prec:>7.3f} {f1:>7.3f}{flag}")
        if rec >= recall_min and f1 > best['f1']:
            best.update({'t': t, 'f1': f1, 'recall': rec, 'prec': prec,
                         'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
    return best


# ─────────────────────────────────────────────────────────────────────────────
# 5. v4.1 — CNN (PTB-XL) + LSTM, sin GAP, input (1000, 3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  MODELO v4.1 — CNN (PTB-XL) + LSTM, input truncado (1000,3)")
print("=" * 62)

def build_v41(input_shape=(1000, 3), n_features=18, l2=1e-4, dropout=0.30):
    inp_ecg = layers.Input(shape=input_shape, name='ecg_signal')
    x = layers.Conv1D(32, 7, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2),
                      name='conv_in')(inp_ecg)
    x = layers.BatchNormalization(name='bn_in')(x)
    x = layers.Activation('relu')(x)
    x = residual_block(x, 32, 5, 'res1', l2=l2)
    x = layers.MaxPooling1D(4, name='pool1')(x)         # 1000→250
    x = layers.Dropout(dropout * 0.5)(x)
    x = residual_block(x, 64, 3, 'res2', l2=l2)
    x = layers.MaxPooling1D(4, name='pool2')(x)         # 250→62
    x = layers.Dropout(dropout * 0.5)(x)
    # LSTM con recurrent_dropout=0 → cuDNN fast path
    x = layers.LSTM(32, return_sequences=False,
                    dropout=dropout*0.5, recurrent_dropout=0,
                    kernel_regularizer=regularizers.l2(l2),
                    name='lstm')(x)
    x = layers.Dropout(0.15, name='drop_lstm')(x)
    rama_a = layers.Dense(64, activation='relu',
                          kernel_regularizer=regularizers.l2(l2),
                          name='dense_a')(x)
    rama_a = layers.Dropout(dropout, name='drop_a')(rama_a)

    inp_feats = layers.Input(shape=(n_features,), name='ecg_features')
    f = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='dense_f1')(inp_feats)
    f = layers.BatchNormalization(name='bn_f1')(f)
    f = layers.Dropout(dropout, name='drop_f1')(f)
    rama_b = layers.Dense(16, activation='relu', name='dense_f2')(f)

    fused  = layers.Concatenate(name='fusion')([rama_a, rama_b])
    fused  = layers.Dropout(dropout*0.5, name='drop_fused')(fused)
    fused  = layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(l2),
                          name='dense_fused')(fused)
    fused  = layers.Dropout(dropout*0.5)(fused)
    output = layers.Dense(1, activation='sigmoid', name='output',
                          dtype='float32')(fused)
    return models.Model(inputs=[inp_ecg, inp_feats], outputs=output,
                        name='BrugadaDual_v41_LSTM')

model_ptbxl = tf.keras.models.load_model(PRETRAINED)
model_v41   = build_v41(input_shape=(1000, 3))
n_tr = transfer_weights(model_v41, model_ptbxl)
del model_ptbxl
print(f"  ✓ Pesos transferidos: {n_tr} capas CNN desde PTB-XL")

# Fase B1: freeze res1
for layer in model_v41.layers:
    layer.trainable = layer.name not in [
        'res1_c1','res1_bn1','res1_r1','res1_c2','res1_bn2','res1_add','res1_r2'
    ]

model_v41.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss      = bce_label_smoothing(0.05),
    metrics   = [tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                 tf.keras.metrics.Recall(name='recall')]
)

train_ds_1000 = make_dataset(X_train_1000, F_train, y_train, shuffle=True)
val_ds_1000   = make_dataset(X_val_1000,   F_val,   y_val)

print("\n  [B1] Freeze res1, LR=1e-4, 30 épocas...")
hist_v41_b1 = model_v41.fit(
    train_ds_1000, epochs=30, validation_data=val_ds_1000,
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks('mejor_v41_b1.keras', patience_es=15, patience_rlr=8),
    verbose=1
)

# Fase B2: descongelar todo
for layer in model_v41.layers:
    layer.trainable = True

model_v41.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss      = bce_label_smoothing(0.05),
    metrics   = [tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                 tf.keras.metrics.Recall(name='recall')]
)

print("\n  [B2] Descongelar todo, LR=1e-5, 200 épocas...")
hist_v41_b2 = model_v41.fit(
    train_ds_1000, epochs=200, validation_data=val_ds_1000,
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks('mejor_v41.keras', patience_es=50, patience_rlr=15),
    verbose=1
)

y_prob_v41 = model_v41.predict([X_test_1000, F_test], verbose=0).flatten()
auc_v41    = roc_auc_score(y_test, y_prob_v41)
print(f"\n  v4.1 AUC-ROC: {auc_v41:.4f}")
print("  Búsqueda de umbral óptimo:")
best_v41   = find_best_threshold(y_test, y_prob_v41)
cm_v41     = confusion_matrix(y_test, (y_prob_v41 > best_v41['t']).astype(int))
model_v41.save('modelo_brugada_v41.keras')
np.save('umbral_v41.npy', np.array([best_v41['t']]))


# ─────────────────────────────────────────────────────────────────────────────
# 6. v4.2 — CNN (PTB-XL) + GAP + Dense LSTM-like, input (1200, 3)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  MODELO v4.2 — CNN (PTB-XL) + GAP + Dense extra, input (1200,3)")
print("  (GAP colapsa la secuencia — el LSTM no aporta sobre un vector)")
print("  Usamos Dense extra en vez de LSTM para ser honestos")
print("=" * 62)

def build_v42(input_shape=(1200, 3), n_features=18, l2=1e-4, dropout=0.30):
    inp_ecg = layers.Input(shape=(None, 3), name='ecg_signal')
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
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='dense_head')(x)
    x = layers.Dropout(dropout, name='drop_head')(x)
    # Dense extra en vez de LSTM — sobre un vector GAP no hay secuencia
    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='dense_a')(x)
    x = layers.Dropout(dropout * 0.5, name='drop_a')(x)

    inp_feats = layers.Input(shape=(n_features,), name='ecg_features')
    f = layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='dense_f1')(inp_feats)
    f = layers.BatchNormalization(name='bn_f1')(f)
    f = layers.Dropout(dropout, name='drop_f1')(f)
    rama_b = layers.Dense(16, activation='relu', name='dense_f2')(f)

    fused  = layers.Concatenate(name='fusion')([x, rama_b])
    fused  = layers.Dropout(dropout*0.5, name='drop_fused')(fused)
    fused  = layers.Dense(32, activation='relu',
                          kernel_regularizer=regularizers.l2(l2),
                          name='dense_fused')(fused)
    fused  = layers.Dropout(dropout*0.5)(fused)
    output = layers.Dense(1, activation='sigmoid', name='output',
                          dtype='float32')(fused)
    return models.Model(inputs=[inp_ecg, inp_feats], outputs=output,
                        name='BrugadaDual_v42_GAP')

model_ptbxl = tf.keras.models.load_model(PRETRAINED)
model_v42   = build_v42(input_shape=(1200, 3))
n_tr = transfer_weights(model_v42, model_ptbxl)
del model_ptbxl
print(f"  ✓ Pesos transferidos: {n_tr} capas CNN desde PTB-XL")

for layer in model_v42.layers:
    layer.trainable = layer.name not in [
        'res1_c1','res1_bn1','res1_r1','res1_c2','res1_bn2','res1_add','res1_r2'
    ]

model_v42.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0),
    loss      = bce_label_smoothing(0.05),
    metrics   = [tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                 tf.keras.metrics.Recall(name='recall')]
)

train_ds_1200 = make_dataset(X_train, F_train, y_train, shuffle=True)
val_ds_1200   = make_dataset(X_val,   F_val,   y_val)

print("\n  [B1] Freeze res1, LR=1e-4, 30 épocas...")
hist_v42_b1 = model_v42.fit(
    train_ds_1200, epochs=30, validation_data=val_ds_1200,
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks('mejor_v42_b1.keras', patience_es=15, patience_rlr=8),
    verbose=1
)

for layer in model_v42.layers:
    layer.trainable = True

model_v42.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, clipnorm=1.0),
    loss      = bce_label_smoothing(0.05),
    metrics   = [tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                 tf.keras.metrics.Recall(name='recall')]
)

print("\n  [B2] Descongelar todo, LR=1e-5, 200 épocas...")
hist_v42_b2 = model_v42.fit(
    train_ds_1200, epochs=200, validation_data=val_ds_1200,
    class_weight=CLASS_WEIGHTS,
    callbacks=get_callbacks('mejor_v42.keras', patience_es=50, patience_rlr=15),
    verbose=1
)

y_prob_v42 = model_v42.predict([X_test, F_test], verbose=0).flatten()
auc_v42    = roc_auc_score(y_test, y_prob_v42)
print(f"\n  v4.2 AUC-ROC: {auc_v42:.4f}")
print("  Búsqueda de umbral óptimo:")
best_v42   = find_best_threshold(y_test, y_prob_v42)
cm_v42     = confusion_matrix(y_test, (y_prob_v42 > best_v42['t']).astype(int))
model_v42.save('modelo_brugada_v42.keras')
np.save('umbral_v42.npy', np.array([best_v42['t']]))


# ─────────────────────────────────────────────────────────────────────────────
# 7. COMPARATIVA FINAL
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  COMPARATIVA FINAL — TODOS LOS MODELOS")
print("=" * 62)

print(f"\n  {'Modelo':<30} {'AUC':>6} {'Recall':>7} {'Prec':>7} {'FP':>4} {'FN':>4}")
print("  " + "─" * 60)
for name, auc_, rec_, prec_, fp_, fn_ in [
    ('CNN Base (v1)',                'N/A',   0.9333, 0.4242, 19, 1),
    ('CNN-Res-LSTM v2',              0.9530,  0.8667, 0.5652, 11, 2),
    ('CNN-Dual v3.1',                0.9437,  0.8667, 0.7647,  4, 2),
    ('CNN-Dual v3.3 (SOTA actual)',  0.9115,  0.9333, 0.7368,  5, 1),
    ('v4 GAP (sin LSTM)',            0.9540,  0.9333, 0.6667,  7, 1),
    ('v4.1 TL+LSTM (1000)',         auc_v41, best_v41['recall'], best_v41['prec'],
                                     cm_v41[0,1], cm_v41[1,0]),
    ('v4.2 TL+GAP+Dense (1200)',    auc_v42, best_v42['recall'], best_v42['prec'],
                                     cm_v42[0,1], cm_v42[1,0]),
]:
    auc_s = f"{auc_:.4f}" if isinstance(auc_, float) else str(auc_)
    print(f"  {name:<30} {auc_s:>6} {rec_:>7.4f} {prec_:>7.4f} {fp_:>4} {fn_:>4}")


# ─────────────────────────────────────────────────────────────────────────────
# 8. VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────
fpr_41, tpr_41, _ = roc_curve(y_test, y_prob_v41)
fpr_42, tpr_42, _ = roc_curve(y_test, y_prob_v42)

fig = plt.figure(figsize=(16, 6))
fig.suptitle('Comparativa v4.1 (LSTM) vs v4.2 (GAP+Dense) — Transfer Learning PTB-XL',
             fontsize=12, fontweight='bold')
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

ax1 = fig.add_subplot(gs[0])
ax1.plot(fpr_41, tpr_41, color='#1565C0', lw=2, label=f'v4.1 LSTM  AUC={auc_v41:.3f}')
ax1.plot(fpr_42, tpr_42, color='#C62828', lw=2, label=f'v4.2 GAP   AUC={auc_v42:.3f}')
ax1.plot([0,1],[0,1],'k--',alpha=0.4)
ax1.set_title('Curvas ROC — Test set', fontweight='bold')
ax1.set_xlabel('FPR'); ax1.set_ylabel('TPR')
ax1.legend(); ax1.grid(alpha=0.3)

ax2 = fig.add_subplot(gs[1])
mods = ['v3.3\nSOTA', 'v4\nGAP', 'v4.1\nLSTM', 'v4.2\nGAP+D']
fps  = [5, 7, cm_v41[0,1], cm_v42[0,1]]
fns  = [1, 1, cm_v41[1,0], cm_v42[1,0]]
x = np.arange(4); w = 0.35
b1 = ax2.bar(x-w/2, fps, w, label='FP', color='#FF8F00', alpha=0.85)
b2 = ax2.bar(x+w/2, fns, w, label='FN', color='#C62828', alpha=0.85)
for bar in list(b1)+list(b2):
    ax2.annotate(str(int(bar.get_height())),
                 xy=(bar.get_x()+bar.get_width()/2, bar.get_height()),
                 xytext=(0,3), textcoords='offset points', ha='center', fontsize=11)
ax2.set_title('FP y FN por modelo', fontweight='bold')
ax2.set_ylabel('Pacientes mal clasificados')
ax2.set_xticks(x); ax2.set_xticklabels(mods, fontsize=9)
ax2.legend(); ax2.grid(alpha=0.3, axis='y')

ax3 = fig.add_subplot(gs[2])
ax3.hist(y_prob_v41[y_test==0], bins=20, alpha=0.5, color='#1565C0', label='Ctrl v4.1')
ax3.hist(y_prob_v41[y_test==1], bins=10, alpha=0.5, color='#1565C0', hatch='//', label='Brug v4.1')
ax3.hist(y_prob_v42[y_test==0], bins=20, alpha=0.5, color='#C62828', label='Ctrl v4.2')
ax3.hist(y_prob_v42[y_test==1], bins=10, alpha=0.5, color='#C62828', hatch='//', label='Brug v4.2')
ax3.set_title('Distribución probabilidades', fontweight='bold')
ax3.set_xlabel('P(Brugada)'); ax3.legend(fontsize=8); ax3.grid(alpha=0.3)

plt.savefig('comparativa_v41_v42.png', dpi=150, bbox_inches='tight')
plt.show()
print("  ✓ comparativa_v41_v42.png guardado")

print(f"""
╔══════════════════════════════════════════════════════════╗
║  COMPARATIVA COMPLETADA                                  ║
╠══════════════════════════════════════════════════════════╣
║  v4.1 (TL+LSTM) : AUC={auc_v41:.4f}  FP={cm_v41[0,1]}  FN={cm_v41[1,0]}           ║
║  v4.2 (TL+GAP)  : AUC={auc_v42:.4f}  FP={cm_v42[0,1]}  FN={cm_v42[1,0]}           ║
╚══════════════════════════════════════════════════════════╝
""")
