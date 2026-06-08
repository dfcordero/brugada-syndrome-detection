import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path

PROCESSED_DIR = Path('ptb_xl_data/processed')
RECORDS_DIR   = PROCESSED_DIR / 'records'
LABELS_CSV    = PROCESSED_DIR / 'labels.csv'
SUPERCLASSES  = ['NORM', 'MI', 'STTC', 'CD', 'HYP']

class PTBXLGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, records_dir, batch_size=64, augment=False, shuffle=True):
        self.records_dir = Path(records_dir)
        self.batch_size  = batch_size
        self.augment     = augment
        self.shuffle     = shuffle
        self.ecg_ids     = df.index.to_numpy()
        self.labels      = df[SUPERCLASSES].values.astype(np.float32)
        self.indices     = np.arange(len(self.ecg_ids))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(len(self.ecg_ids) / self.batch_size))

    def __getitem__(self, batch_idx):
        start         = batch_idx * self.batch_size
        end           = min(start + self.batch_size, len(self.ecg_ids))
        batch_indices = self.indices[start:end]
        X_batch = np.zeros((len(batch_indices), 1000, 3), dtype=np.float32)
        y_batch = self.labels[batch_indices]
        for i, idx in enumerate(batch_indices):
            npy_path = self.records_dir / f"{self.ecg_ids[idx]}.npy"
            try:
                signal = np.load(npy_path)
                if self.augment:
                    signal = self._augment(signal)
                X_batch[i] = signal
            except Exception:
                pass
        return X_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _augment(self, signal):
        aug = signal.copy()
        if np.random.random() < 0.5:
            aug += np.random.normal(0, 0.01, aug.shape).astype(np.float32)
        if np.random.random() < 0.5:
            aug *= np.random.uniform(0.90, 1.10)
        if np.random.random() < 0.3:
            aug = np.roll(aug, np.random.randint(-20, 20), axis=0)
        return aug.astype(np.float32)

def create_ptbxl_splits(labels_csv=LABELS_CSV, val_fold=9, test_fold=10):
    df       = pd.read_csv(labels_csv, index_col='ecg_id')
    df_train = df[~df['strat_fold'].isin([val_fold, test_fold])].drop(columns='strat_fold')
    df_val   = df[df['strat_fold'] == val_fold].drop(columns='strat_fold')
    df_test  = df[df['strat_fold'] == test_fold].drop(columns='strat_fold')
    print(f"  Train: {len(df_train)}  Val: {len(df_val)}  Test: {len(df_test)}")
    return df_train, df_val, df_test

if __name__ == '__main__':
    import time
    print("=" * 62)
    print("  TEST DEL DATA GENERATOR")
    print("=" * 62)
    print(f"  Buscando: {LABELS_CSV.resolve()}")
    if not LABELS_CSV.exists():
        print(f"  [ERROR] No encontrado")
        exit()
    df_train, df_val, df_test = create_ptbxl_splits()
    train_gen = PTBXLGenerator(df_train, RECORDS_DIR, batch_size=64, augment=True, shuffle=True)
    val_gen   = PTBXLGenerator(df_val,   RECORDS_DIR, batch_size=64, augment=False, shuffle=False)
    print(f"  Batches/época — train: {len(train_gen)}  val: {len(val_gen)}")
    t0   = time.time()
    X, y = train_gen[0]
    ms   = (time.time() - t0) * 1000
    print(f"  X shape: {X.shape}  y shape: {y.shape}")
    print(f"  Tiempo : {ms:.1f}ms por batch")
    print(f"  Rango X: [{X.min():.3f}, {X.max():.3f}]")
    print("  TEST OK — siguiente: fase3A_3_pretrain_ptbxl.py")
