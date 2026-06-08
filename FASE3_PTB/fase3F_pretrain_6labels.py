"""
================================================================================
  FASE 3F — PREENTRENAMIENTO 6 CLASES (PTB-XL + Brugada)
================================================================================
  Extiende el preentrenamiento PTB-XL añadiendo Brugada como 6.ª etiqueta.
  La arquitectura es idéntica a fase3A_3_pretrain_ptbxl.py pero con
  Dense(6) en la salida; los pesos del modelo de 5 clases se transfieren
  capa a capa (by name), saltando solo la capa 'output' (5→6 neuronas).

  Clases de salida (índices 0-5):
    0: NORM  1: MI  2: STTC  3: CD  4: HYP  5: BRUGADA

  Ejecutar desde: ~/proyectos/brugada_syndrom/FASE3_PTB/
  python fase3F_pretrain_6labels.py

  Salidas:
    ptb_xl_data/processed/labels_6class.csv
    mejor_modelo_pretrained_6class.keras
    pretrain_6class_results.png
================================================================================
"""

# ── GPU ───────────────────────────────────────────────────────────────────────
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import layers, models, callbacks, regularizers
from sklearn.metrics import roc_auc_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ─── Rutas ────────────────────────────────────────────────────────────────────
BASE_DIR      = Path('.')
RECORDS_DIR   = BASE_DIR / 'ptb_xl_data/processed/records'
LABELS_CSV    = BASE_DIR / 'ptb_xl_data/processed/labels.csv'
LABELS_6CLASS = BASE_DIR / 'ptb_xl_data/processed/labels_6class.csv'
PRETRAIN_5CLS = BASE_DIR / 'mejor_modelo_pretrained_ptbxl.keras'
MODEL_OUT     = BASE_DIR / 'mejor_modelo_pretrained_6class.keras'
FIG_OUT       = BASE_DIR / 'pretrain_6class_results.png'

FASE2_DIR     = BASE_DIR / '../FASE2'

# ─── Hiperparámetros ──────────────────────────────────────────────────────────
BATCH_SIZE = 64
EPOCHS     = 50
LR         = 3e-4
L2_REG     = 1e-4
DROPOUT    = 0.30
SUPERCLASSES_6 = ['NORM', 'MI', 'STTC', 'CD', 'HYP', 'BRUGADA']


# ─── PASO 1: Construir labels_6class.csv ─────────────────────────────────────

def build_6class_labels(ptbxl_csv, brugada_splits):
    """
    Combina las etiquetas PTB-XL (5 clases) con los registros Brugada (6.ª clase).

    brugada_splits: dict con claves 'train'/'val'/'test' y valores array (N, 1000, 3)
    strat_fold asignado: train=8, val=9, test=10  (consistente con PTB-XL)
    """
    df_ptb = pd.read_csv(ptbxl_csv)
    df_ptb['BRUGADA'] = 0

    brugada_rows = []
    fold_map = {'train': 8, 'val': 9, 'test': 10}
    ecg_id_counter = 100000  # IDs ficticios para Brugada (> max PTB-XL ~21k)
    for split_name, signals in brugada_splits.items():
        n = len(signals)
        for _ in range(n):
            brugada_rows.append({
                'ecg_id':     ecg_id_counter,
                'strat_fold': fold_map[split_name],
                'NORM': 0, 'MI': 0, 'STTC': 0, 'CD': 0, 'HYP': 0,
                'BRUGADA': 1,
            })
            ecg_id_counter += 1

    df_brugada = pd.DataFrame(brugada_rows)
    df_combined = pd.concat([df_ptb, df_brugada], ignore_index=True)
    df_combined.to_csv(LABELS_6CLASS, index=False)
    print(f"  labels_6class.csv guardado: {len(df_combined)} filas")

    n_ptb_train = len(df_ptb[~df_ptb['strat_fold'].isin([9, 10])])
    n_brug_train = len(brugada_splits['train'])
    print(f"  PTB-XL train: {n_ptb_train}  Brugada train: {n_brug_train}")
    brugada_weight = n_ptb_train / max(n_brug_train, 1)
    print(f"  Brugada sample weight: {brugada_weight:.1f}x")

    return df_combined, brugada_weight


# ─── PASO 2: CombinedGenerator ────────────────────────────────────────────────

class CombinedGenerator(tf.keras.utils.Sequence):
    """
    Genera batches mezclando registros PTB-XL (disco) y señales Brugada (memoria).

    - PTB-XL: se carga por .npy desde RECORDS_DIR en cada batch.
    - Brugada: array numpy precargado en memoria.
    - Las muestras Brugada reciben sample_weight = brugada_weight.
    - PTB-XL recibe sample_weight = 1.0.
    """

    def __init__(self, df_ptb, brugada_signals, records_dir,
                 batch_size=64, augment=False, shuffle=True,
                 brugada_weight=1.0):
        """
        df_ptb          : DataFrame con columnas [NORM, MI, STTC, CD, HYP],
                          index = ecg_id (enteros PTB-XL).
        brugada_signals : ndarray (N_brug, 1000, 3) float32.
        """
        self.records_dir     = Path(records_dir)
        self.batch_size      = batch_size
        self.augment         = augment
        self.shuffle         = shuffle
        self.brugada_weight  = float(brugada_weight)

        # PTB-XL
        self.ptb_ids    = df_ptb.index.to_numpy()
        ptb_labels_5    = df_ptb[['NORM', 'MI', 'STTC', 'CD', 'HYP']].values.astype(np.float32)
        # Añadir columna BRUGADA=0
        self.ptb_labels = np.hstack([ptb_labels_5,
                                     np.zeros((len(ptb_labels_5), 1), dtype=np.float32)])

        # Brugada
        self.brug_signals = brugada_signals.astype(np.float32)  # (N, 1000, 3)
        n_brug = len(self.brug_signals)
        brug_labels_5 = np.zeros((n_brug, 5), dtype=np.float32)
        brug_label_6  = np.ones((n_brug, 1), dtype=np.float32)
        self.brug_labels = np.hstack([brug_labels_5, brug_label_6])

        # Índices combinados: 0..N_ptb-1 → PTB-XL, N_ptb..N_ptb+N_brug-1 → Brugada
        self.n_ptb  = len(self.ptb_ids)
        self.n_brug = n_brug
        self.n_total = self.n_ptb + self.n_brug

        self.indices = np.arange(self.n_total)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.n_total / self.batch_size))

    def __getitem__(self, batch_idx):
        start = batch_idx * self.batch_size
        end   = min(start + self.batch_size, self.n_total)
        idx   = self.indices[start:end]
        bs    = len(idx)

        X_batch = np.zeros((bs, 1000, 3), dtype=np.float32)
        y_batch = np.zeros((bs, 6), dtype=np.float32)
        w_batch = np.ones(bs, dtype=np.float32)

        for i, global_idx in enumerate(idx):
            if global_idx < self.n_ptb:
                # PTB-XL record — cargar de disco
                eid  = self.ptb_ids[global_idx]
                path = self.records_dir / f"{eid}.npy"
                try:
                    sig = np.load(path).astype(np.float32)
                    if self.augment:
                        sig = self._augment(sig)
                    X_batch[i] = sig
                except Exception:
                    pass
                y_batch[i] = self.ptb_labels[global_idx]
                w_batch[i] = 1.0
            else:
                # Brugada — en memoria
                brug_idx   = global_idx - self.n_ptb
                sig = self.brug_signals[brug_idx].copy()
                if self.augment:
                    sig = self._augment(sig)
                X_batch[i] = sig
                y_batch[i] = self.brug_labels[brug_idx]
                w_batch[i] = self.brugada_weight

        return X_batch, y_batch, w_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    @staticmethod
    def _augment(signal):
        aug = signal.copy()
        if np.random.random() < 0.5:
            aug += np.random.normal(0, 0.01, aug.shape).astype(np.float32)
        if np.random.random() < 0.5:
            aug *= np.random.uniform(0.90, 1.10)
        if np.random.random() < 0.3:
            aug = np.roll(aug, np.random.randint(-20, 20), axis=0)
        return aug.astype(np.float32)


# ─── PASO 3: Arquitectura ─────────────────────────────────────────────────────

def residual_block(x, filters, kernel_size, name, l2=L2_REG):
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


def build_6class_model():
    """
    Arquitectura idéntica a fase3A_3_pretrain_ptbxl.py con Dense(6) en la salida.
    Nombres de capa idénticos para facilitar la transferencia de pesos.
    """
    inp = layers.Input(shape=(None, 3), name='ecg_signal')

    x = layers.Conv1D(32, 7, padding='same', use_bias=False,
                      kernel_regularizer=regularizers.l2(L2_REG),
                      name='conv_in')(inp)
    x = layers.BatchNormalization(name='bn_in')(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, 32, 5, 'res1')
    x = layers.MaxPooling1D(4, name='pool1')(x)
    x = layers.Dropout(DROPOUT * 0.5)(x)

    x = residual_block(x, 64, 3, 'res2')
    x = layers.MaxPooling1D(4, name='pool2')(x)
    x = layers.Dropout(DROPOUT * 0.5)(x)

    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(L2_REG),
                     name='dense_head')(x)
    x = layers.Dropout(DROPOUT, name='drop_head')(x)

    # 6 clases: NORM, MI, STTC, CD, HYP, BRUGADA
    output = layers.Dense(6, activation='sigmoid',
                          name='output', dtype='float32')(x)

    return models.Model(inputs=inp, outputs=output, name='PTBXLPretrain6Class')


# ─── PASO 4: Transferencia de pesos ──────────────────────────────────────────

def transfer_weights_from_5class(new_model, pretrain_path):
    """
    Carga el modelo de 5 clases y transfiere pesos capa a capa por nombre.
    La capa 'output' se salta (5→6 neuronas incompatibles).
    """
    print(f"\n  Cargando modelo base: {pretrain_path} …")
    old_model = tf.keras.models.load_model(str(pretrain_path))

    transferred, skipped = [], []
    for layer in new_model.layers:
        try:
            old_layer = old_model.get_layer(layer.name)
            old_weights = old_layer.get_weights()
            new_weights = layer.get_weights()
            if len(old_weights) == 0:
                continue
            # Verificar compatibilidad de shapes
            if all(o.shape == n.shape for o, n in zip(old_weights, new_weights)):
                layer.set_weights(old_weights)
                transferred.append(layer.name)
            else:
                skipped.append(f"{layer.name} (shape mismatch)")
        except ValueError:
            skipped.append(f"{layer.name} (not in source)")

    print(f"  Capas transferidas ({len(transferred)}): {transferred}")
    print(f"  Capas saltadas     ({len(skipped)}): {skipped}")
    return new_model


# ─── PASO 5: Evaluación ───────────────────────────────────────────────────────

def evaluate_6class(model, gen, label='Test'):
    """
    Evalúa AUC-ROC por clase y muestra confusion matrix de Brugada a thr=0.5.
    """
    y_true_all, y_pred_all = [], []
    for i in range(len(gen)):
        batch = gen[i]
        X_b, y_b = batch[0], batch[1]
        y_pred_all.append(model.predict(X_b, verbose=0))
        y_true_all.append(y_b)

    y_true = np.vstack(y_true_all)
    y_pred = np.vstack(y_pred_all)

    print(f"\n  AUC-ROC por clase ({label}):")
    aucs = {}
    for i, cls in enumerate(SUPERCLASSES_6):
        n_pos = int(y_true[:, i].sum())
        if n_pos > 0:
            auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            aucs[cls] = auc
            marker = ' ◄' if cls == 'BRUGADA' else ''
            print(f"    {cls:<8}: {auc:.4f}  (n_pos={n_pos}){marker}")
        else:
            print(f"    {cls:<8}: N/A (sin positivos en {label})")

    macro = np.mean(list(aucs.values())) if aucs else 0.0
    print(f"    {'Macro':<8}: {macro:.4f}")

    # Confusion matrix Brugada @ thr=0.5
    brug_true = y_true[:, 5].astype(int)
    brug_pred = (y_pred[:, 5] >= 0.5).astype(int)
    if brug_true.sum() > 0:
        tn, fp, fn, tp = confusion_matrix(brug_true, brug_pred, labels=[0, 1]).ravel()
        print(f"\n  Confusion matrix BRUGADA @ thr=0.5 ({label}):")
        print(f"    TP={tp}  FP={fp}  TN={tn}  FN={fn}")
        print(f"    Recall  = {tp / max(tp+fn, 1):.3f}")
        print(f"    Precision = {tp / max(tp+fp, 1):.3f}")

    return aucs, macro, y_pred, y_true


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 66)
    print("  FASE 3F — PREENTRENAMIENTO 6 CLASES (PTB-XL + Brugada)")
    print("=" * 66)

    # ── Cargar señales Brugada reales (sin aumentación) ───────────────────────
    print("\n[1] Cargando señales Brugada …")
    brugada_splits = {}
    total_brugada = 0
    for split, fold in [('train', 8), ('val', 9), ('test', 10)]:
        X = np.load(str(FASE2_DIR / f'X_{split}.npy'))
        y = np.load(str(FASE2_DIR / f'y_{split}.npy'))
        X_brug = X[y == 1].astype(np.float32)
        # Truncar a (1000, 3) — las señales Brugada son (1200, 3)
        if X_brug.shape[1] > 1000:
            X_brug = X_brug[:, :1000, :3]
        elif X_brug.shape[1] < 1000:
            # Padding si fuese necesario (caso improbable)
            pad = np.zeros((len(X_brug), 1000 - X_brug.shape[1], X_brug.shape[2]),
                           dtype=np.float32)
            X_brug = np.concatenate([X_brug, pad], axis=1)
        else:
            X_brug = X_brug[:, :, :3]
        brugada_splits[split] = X_brug
        total_brugada += len(X_brug)
        print(f"    {split}: {len(X_brug)} Brugada  (fold={fold})")

    print(f"    Total Brugada: {total_brugada}")

    # ── PASO 1: Crear labels_6class.csv ───────────────────────────────────────
    print("\n[2] Construyendo labels_6class.csv …")
    df_combined, brug_weight = build_6class_labels(LABELS_CSV, brugada_splits)

    # Splits usando strat_fold (idéntica lógica a create_ptbxl_splits)
    df_train_full = df_combined[~df_combined['strat_fold'].isin([9, 10])].copy()
    df_val_full   = df_combined[df_combined['strat_fold'] == 9].copy()
    df_test_full  = df_combined[df_combined['strat_fold'] == 10].copy()

    # Separar PTB-XL (ecg_id < 100000) de Brugada (ecg_id >= 100000)
    def split_ptb_brug(df):
        ptb_mask  = df['ecg_id'] < 100000
        df_ptb    = df[ptb_mask].set_index('ecg_id')
        return df_ptb

    df_ptb_train = split_ptb_brug(df_train_full)
    df_ptb_val   = split_ptb_brug(df_val_full)
    df_ptb_test  = split_ptb_brug(df_test_full)

    print(f"\n    PTB-XL  — train: {len(df_ptb_train)}  val: {len(df_ptb_val)}  test: {len(df_ptb_test)}")
    print(f"    Brugada — train: {len(brugada_splits['train'])}  "
          f"val: {len(brugada_splits['val'])}  test: {len(brugada_splits['test'])}")

    # ── PASO 2: Generadores ───────────────────────────────────────────────────
    print("\n[3] Creando generadores combinados …")
    train_gen = CombinedGenerator(
        df_ptb_train, brugada_splits['train'], RECORDS_DIR,
        batch_size=BATCH_SIZE, augment=True, shuffle=True,
        brugada_weight=brug_weight
    )
    val_gen = CombinedGenerator(
        df_ptb_val, brugada_splits['val'], RECORDS_DIR,
        batch_size=BATCH_SIZE, augment=False, shuffle=False,
        brugada_weight=1.0  # val sin upweighting para métricas reales
    )
    test_gen = CombinedGenerator(
        df_ptb_test, brugada_splits['test'], RECORDS_DIR,
        batch_size=BATCH_SIZE, augment=False, shuffle=False,
        brugada_weight=1.0
    )

    print(f"    Batches/época — train: {len(train_gen)}  val: {len(val_gen)}")
    print(f"    Muestras total — train: {train_gen.n_total}  val: {val_gen.n_total}  test: {test_gen.n_total}")

    # ── PASO 3 & 4: Modelo y transferencia de pesos ───────────────────────────
    print("\n[4] Construyendo modelo 6 clases …")
    model = build_6class_model()

    print("\n[5] Transfiriendo pesos del modelo de 5 clases …")
    model = transfer_weights_from_5class(model, PRETRAIN_5CLS)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss='binary_crossentropy',
        metrics=[
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.BinaryAccuracy(name='acc'),
        ]
    )
    model.summary()
    print(f"\n  Parámetros totales: {model.count_params():,}")
    print(f"  Salida: (None, 6)  →  {SUPERCLASSES_6}")

    # ── Evaluación baseline (antes de entrenar) ───────────────────────────────
    print("\n[6] Evaluación baseline (pesos transferidos, sin fine-tuning) …")
    evaluate_6class(model, test_gen, label='Test (baseline)')

    # ── PASO 4: Entrenamiento ─────────────────────────────────────────────────
    print(f"\n[7] Entrenamiento — {EPOCHS} épocas máx, LR={LR}, batch={BATCH_SIZE}")
    print("=" * 66)

    cbs = [
        callbacks.EarlyStopping(
            monitor='val_loss', patience=8,
            mode='min', restore_best_weights=True, verbose=1
        ),
        callbacks.ModelCheckpoint(
            filepath=str(MODEL_OUT), monitor='val_loss',
            save_best_only=True, mode='min', verbose=0
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
        callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: print(
                f"  [ep {epoch+1:>3}] "
                f"loss={logs.get('loss', 0):.4f} | "
                f"val_loss={logs.get('val_loss', 0):.4f} | "
                f"val_auc={logs.get('val_auc', 0):.3f}"
            ) if (epoch + 1) % 5 == 0 else None
        ),
    ]

    history = model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen,
        callbacks=cbs,
        verbose=1
    )

    best_epoch = int(np.argmin(history.history['val_loss'])) + 1
    print(f"\n  Mejor época: {best_epoch}")

    # ── PASO 5: Evaluación final ──────────────────────────────────────────────
    print("\n[8] Evaluación final en test set …")
    aucs, macro_auc, y_pred_test, y_true_test = evaluate_6class(
        model, test_gen, label='Test'
    )

    # ── PASO 6: Figura ────────────────────────────────────────────────────────
    print("\n[9] Generando figura …")
    ep = range(1, len(history.history['loss']) + 1)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(
        'Preentrenamiento 6 Clases — PTB-XL + Brugada\n'
        'CNN Residual + GAP  |  Transfer Learning desde modelo 5 clases',
        fontweight='bold', fontsize=13
    )

    gs = plt.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Loss curves
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, history.history['loss'],     label='Train', color='#1565C0', lw=1.5)
    ax1.plot(ep, history.history['val_loss'], label='Val',   color='#C62828', lw=1.5, ls='--')
    ax1.axvline(best_epoch, color='gray', ls=':', alpha=0.7, label=f'Best ep {best_epoch}')
    ax1.set_title('Loss (binary crossentropy)', fontweight='bold')
    ax1.set_xlabel('Época'); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax1.set_facecolor('#F9F9F9')

    # 2. AUC curves
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, history.history['auc'],     label='Train AUC', color='#6A1B9A', lw=1.5)
    ax2.plot(ep, history.history['val_auc'], label='Val AUC',   color='#6A1B9A', lw=1.5, ls='--')
    ax2.axhline(0.90, color='gray', ls=':', alpha=0.6, label='Ref 0.90')
    ax2.set_title('AUC-ROC multi-label', fontweight='bold')
    ax2.set_xlabel('Época'); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    ax2.set_ylim([0.5, 1.02]); ax2.set_facecolor('#F9F9F9')

    # 3. AUC por clase (barplot)
    ax3 = fig.add_subplot(gs[0, 2])
    cls_names = list(aucs.keys())
    cls_aucs  = list(aucs.values())
    colors = ['#2196F3' if c != 'BRUGADA' else '#E53935' for c in cls_names]
    bars = ax3.bar(cls_names, cls_aucs, color=colors, alpha=0.85)
    for bar, v in zip(bars, cls_aucs):
        ax3.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                 f'{v:.3f}', ha='center', va='bottom', fontsize=8)
    ax3.axhline(macro_auc, color='black', ls='--', lw=1, alpha=0.6,
                label=f'Macro={macro_auc:.3f}')
    ax3.set_ylim([0, 1.08]); ax3.set_title('AUC por clase (Test)', fontweight='bold')
    ax3.set_ylabel('AUC-ROC'); ax3.legend(fontsize=8); ax3.grid(axis='y', alpha=0.3)
    ax3.set_facecolor('#F9F9F9')
    plt.setp(ax3.get_xticklabels(), rotation=30, ha='right', fontsize=8)

    # 4. Distribución probabilidad BRUGADA en test
    ax4 = fig.add_subplot(gs[1, 0])
    brug_true = y_true_test[:, 5].astype(int)
    brug_prob = y_pred_test[:, 5]
    ax4.hist(brug_prob[brug_true == 0], bins=30, alpha=0.6, color='steelblue',
             label='No Brugada (PTB-XL)', density=True)
    ax4.hist(brug_prob[brug_true == 1], bins=20, alpha=0.7, color='crimson',
             label='Brugada', density=True)
    ax4.axvline(0.5, color='black', ls='--', lw=1.2, label='thr=0.5')
    ax4.set_title('P(Brugada) en test', fontweight='bold')
    ax4.set_xlabel('Probabilidad predicha'); ax4.set_ylabel('Densidad')
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3); ax4.set_facecolor('#F9F9F9')

    # 5. Tabla resumen
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')

    brug_prob_pos = brug_prob[brug_true == 1]
    brug_prob_neg = brug_prob[brug_true == 0]

    table_rows = []
    for cls in SUPERCLASSES_6:
        i = SUPERCLASSES_6.index(cls)
        n_pos = int(y_true_test[:, i].sum())
        n_neg = int((y_true_test[:, i] == 0).sum())
        auc_val = f"{aucs[cls]:.4f}" if cls in aucs else "N/A"
        fp_rate = (((y_pred_test[:, i] >= 0.5) & (y_true_test[:, i] == 0)).sum()
                   / max(n_neg, 1) * 100)
        table_rows.append([cls, str(n_pos), str(n_neg), auc_val, f"{fp_rate:.1f}%"])

    col_labels = ['Clase', 'N+', 'N−', 'AUC-ROC', 'FP@0.5']
    table = ax5.table(cellText=table_rows, colLabels=col_labels,
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)
    for j in range(len(col_labels)):
        table[(0, j)].set_facecolor('#1A237E')
        table[(0, j)].set_text_props(color='white', fontweight='bold')
    for i, row in enumerate(table_rows, start=1):
        bg = '#FFEBEE' if row[0] == 'BRUGADA' else ('#F3F4F6' if i % 2 == 0 else 'white')
        for j in range(len(col_labels)):
            table[(i, j)].set_facecolor(bg)

    ax5.set_title('Resumen por clase — Test set', fontweight='bold', pad=12)

    plt.savefig(str(FIG_OUT), dpi=150, bbox_inches='tight')
    print(f"  Figura guardada: {FIG_OUT}")

    model.save(str(MODEL_OUT))
    print(f"  Modelo guardado: {MODEL_OUT}")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  FASE 3F COMPLETADA                                          ║
╠══════════════════════════════════════════════════════════════╣
║  Mejor época      : {best_epoch:<5}                                    ║
║  AUC macro test   : {macro_auc:.4f}                                ║
║  AUC BRUGADA test : {aucs.get('BRUGADA', 0):.4f}                                ║
║  Modelo guardado  : mejor_modelo_pretrained_6class.keras     ║
║                                                              ║
║  SIGUIENTE: usar este modelo como base en fase3B/fase3G      ║
╚══════════════════════════════════════════════════════════════╝
    """)


if __name__ == '__main__':
    main()
