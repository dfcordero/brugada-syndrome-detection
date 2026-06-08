"""
================================================================================
  TRANSFER LEARNING — FASE A, SCRIPT 3
  Preentrenamiento CNN en PTB-XL — arquitectura agnóstica a longitud
================================================================================
  Ejecutar desde: ~/proyectos/brugada_syndrom/FASE3_PTB/
  python fase3A_3_pretrain_ptbxl.py

  Arquitectura:
    CNN Residual (res1 + res2) → GlobalAveragePooling1D → Dense(5, sigmoid)

  Por qué GAP en vez de LSTM:
    GlobalAveragePooling1D colapsa (T, 64) → (64,) para cualquier T.
    PTB-XL entra como (1000, 3) y Brugada entrará como (1200, 3).
    El mismo modelo acepta ambas longitudes sin cambiar nada.

  Output:
    mejor_modelo_pretrained_ptbxl.keras  — listo para Fase B (fine-tuning)
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
else:
    print("  Sin GPU — usando CPU")

import numpy as np
from pathlib import Path
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

from fase3A_2_data_generator import (
    PTBXLGenerator, create_ptbxl_splits, SUPERCLASSES, RECORDS_DIR
)

import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 3e-4
L2_REG     = 1e-4
DROPOUT    = 0.30


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITECTURA
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


def build_pretrain_model(n_classes=5, l2=L2_REG, dropout=DROPOUT):
    """
    CNN Residual + GlobalAveragePooling1D + cabeza multi-label.

    La clave es GlobalAveragePooling1D:
      - PTB-XL:  input (None, 1000, 3) → conv → (None, 1000, 64) → GAP → (None, 64)
      - Brugada: input (None, 1200, 3) → conv → (None, 1200, 64) → GAP → (None, 64)
    El mismo modelo acepta ambas longitudes sin padding ni truncado.

    Nombres de capas idénticos al modelo Brugada para facilitar
    la transferencia de pesos en la Fase B.
    """
    # Input — None permite cualquier longitud temporal
    inp = layers.Input(shape=(None, 3), name='ecg_signal')

    # Capa de entrada
    x = layers.Conv1D(32, 7, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(l2),
                      name='conv_in')(inp)
    x = layers.BatchNormalization(name='bn_in')(x)
    x = layers.Activation('relu')(x)

    # Bloque residual 1 — features básicos (pendientes, picos, duración QRS)
    x = residual_block(x, 32, 5, 'res1', l2=l2)
    x = layers.MaxPooling1D(4, name='pool1')(x)       # 1000→250 / 1200→300
    x = layers.Dropout(dropout * 0.5)(x)

    # Bloque residual 2 — features complejos (morfología ST, onda T, P)
    x = residual_block(x, 64, 3, 'res2', l2=l2)
    x = layers.MaxPooling1D(4, name='pool2')(x)       # 250→62  / 300→75
    x = layers.Dropout(dropout * 0.5)(x)

    # GlobalAveragePooling1D — colapsa dimensión temporal
    # (batch, T, 64) → (batch, 64) independiente de T
    x = layers.GlobalAveragePooling1D(name='gap')(x)

    # Cabeza de clasificación
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='dense_head')(x)
    x = layers.Dropout(dropout, name='drop_head')(x)

    # Salida multi-label: 5 neuronas sigmoid independientes
    # binary_crossentropy trata cada clase de forma independiente
    output = layers.Dense(n_classes, activation='sigmoid',
                          name='output', dtype='float32')(x)

    model = models.Model(inputs=inp, outputs=output, name='PTBXLPretrain')
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def get_callbacks(path='mejor_modelo_pretrained_ptbxl.keras'):
    return [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=8,
            mode='min', restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=path, monitor='val_loss',
            save_best_only=True, mode='min', verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
        callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"  [ep {epoch+1:>3}] "
                f"loss={logs['loss']:.4f} | "
                f"val_loss={logs['val_loss']:.4f} | "
                f"val_auc={logs.get('val_auc', 0):.3f}"
            ) if (epoch+1) % 5 == 0 else None
        ),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# EVALUACIÓN MULTI-LABEL
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_multilabel(model, gen, name='Test'):
    """Calcula AUC-ROC por clase y macro-promedio."""
    y_true_all = []
    y_pred_all = []

    for i in range(len(gen)):
        X_batch, y_batch = gen[i]
        y_pred = model.predict(X_batch, verbose=0)
        y_true_all.append(y_batch)
        y_pred_all.append(y_pred)

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    print(f"\n  AUC-ROC por clase ({name}):")
    aucs = []
    for i, sc in enumerate(SUPERCLASSES):
        if y_true[:, i].sum() > 0:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs.append(auc)
            print(f"    {sc:<6}: {auc:.4f}")
        else:
            print(f"    {sc:<6}: N/A (sin positivos)")

    print(f"    Macro : {np.mean(aucs):.4f}")
    return np.mean(aucs)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':

    print("\n" + "=" * 62)
    print("  FASE A — SCRIPT 3: PREENTRENAMIENTO EN PTB-XL")
    print("=" * 62)

    # Splits
    print("\n  Creando splits...")
    df_train, df_val, df_test = create_ptbxl_splits()

    train_gen = PTBXLGenerator(df_train, RECORDS_DIR, batch_size=BATCH_SIZE,
                               augment=True, shuffle=True)
    val_gen   = PTBXLGenerator(df_val,   RECORDS_DIR, batch_size=BATCH_SIZE,
                               augment=False, shuffle=False)
    test_gen  = PTBXLGenerator(df_test,  RECORDS_DIR, batch_size=BATCH_SIZE,
                               augment=False, shuffle=False)

    print(f"\n  Batches/época — train: {len(train_gen)}  val: {len(val_gen)}")

    # Modelo
    print("\n" + "=" * 62)
    print("  ARQUITECTURA")
    print("=" * 62)

    model = build_pretrain_model()
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss      = 'binary_crossentropy',
        metrics   = [
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.BinaryAccuracy(name='acc'),
        ]
    )
    model.summary()

    total_params = model.count_params()
    print(f"\n  Parámetros totales: {total_params:,}")
    print(f"  Input shape       : (None, None, 3) — agnóstico a longitud")
    print(f"  Output shape      : (None, 5) — multi-label sigmoid")

    # Entrenamiento
    print("\n" + "=" * 62)
    print(f"  ENTRENAMIENTO — {EPOCHS} épocas máx, LR={LR}, batch={BATCH_SIZE}")
    print("=" * 62)

    history = model.fit(
        train_gen,
        epochs          = EPOCHS,
        validation_data = val_gen,
        callbacks       = get_callbacks('mejor_modelo_pretrained_ptbxl.keras'),
        verbose         = 1
    )

    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    print(f"\n  Mejor época: {best_epoch}")

    # Evaluación final
    print("\n" + "=" * 62)
    print("  EVALUACIÓN FINAL EN TEST SET")
    print("=" * 62)
    macro_auc = evaluate_multilabel(model, test_gen, name='Test')

    # Curvas de entrenamiento
    ep = range(1, len(history.history['loss']) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Preentrenamiento PTB-XL — CNN Residual + GAP', fontweight='bold')

    axes[0].plot(ep, history.history['loss'],     label='Train', color='#1565C0')
    axes[0].plot(ep, history.history['val_loss'], label='Val',   color='#C62828', ls='--')
    axes[0].axvline(best_epoch, color='gray', ls=':', alpha=0.6)
    axes[0].set_title('Loss (binary crossentropy)', fontweight='bold')
    axes[0].set_xlabel('Época'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(ep, history.history['auc'],     label='Train', color='#6A1B9A')
    axes[1].plot(ep, history.history['val_auc'], label='Val',   color='#6A1B9A', ls='--')
    axes[1].axhline(0.90, color='gray', ls=':', alpha=0.6, label='Ref 0.90')
    axes[1].set_title('AUC-ROC multi-label', fontweight='bold')
    axes[1].set_xlabel('Época'); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[1].set_ylim([0.5, 1.02])

    plt.tight_layout()
    plt.savefig('resultados_pretrain_ptbxl.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("  ✓ resultados_pretrain_ptbxl.png guardado")

    # Guardar modelo
    model.save('mejor_modelo_pretrained_ptbxl.keras')

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  FASE A COMPLETADA — MODELO LISTO PARA FASE B            ║
╠══════════════════════════════════════════════════════════╣
║  Mejor época    : {best_epoch}                                    ║
║  AUC macro test : {macro_auc:.4f}                             ║
║  Modelo guardado: mejor_modelo_pretrained_ptbxl.keras    ║
║                                                          ║
║  SIGUIENTE: fase3B_finetune_brugada.py                   ║
╚══════════════════════════════════════════════════════════╝
    """)
