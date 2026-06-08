"""
================================================================================
  TRANSFER LEARNING — FASE A, SCRIPT 1
  Procesa PTB-XL: extrae V1/V2/V3, etiqueta 5 superclases, guarda .npy
================================================================================
  Prerequisito: dataset descargado desde Kaggle en:
    ptb_xl_data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1/

  Output:
    ptb_xl_data/processed/records/   → archivos .npy (1000, 3) por paciente
    ptb_xl_data/processed/labels.csv → ecg_id, strat_fold, NORM, MI, STTC, CD, HYP
================================================================================
"""

import ast
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# RUTAS — ajustadas a la estructura real de Kaggle
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = Path('ptb_xl_data/raw/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1')
PROCESSED_DIR = Path('ptb_xl_data/processed')
RECORDS_DIR   = PROCESSED_DIR / 'records'

TARGET_LEADS = ['V1', 'V2', 'V3']

SUPERCLASSES = {
    'NORM': 0,
    'MI':   1,
    'STTC': 2,
    'CD':   3,
    'HYP':  4,
}


# ─────────────────────────────────────────────────────────────────────────────
# 1. PARSEAR ETIQUETAS
# ─────────────────────────────────────────────────────────────────────────────
def parse_labels(base_dir: Path) -> pd.DataFrame:
    csv_path  = base_dir / 'ptbxl_database.csv'
    stmt_path = base_dir / 'scp_statements.csv'

    df    = pd.read_csv(csv_path, index_col='ecg_id')
    stmts = pd.read_csv(stmt_path, index_col=0)

    # Solo códigos diagnósticos
    stmts = stmts[stmts['diagnostic'] == 1]
    code_to_super = stmts['diagnostic_class'].to_dict()

    # Parsear scp_codes
    df['scp_codes'] = df['scp_codes'].apply(ast.literal_eval)

    # Inicializar columnas de superclases
    for sc in SUPERCLASSES:
        df[sc] = 0

    for idx, row in df.iterrows():
        for code, prob in row['scp_codes'].items():
            if prob >= 50 and code in code_to_super:
                sc = code_to_super[code]
                if sc in SUPERCLASSES:
                    df.at[idx, sc] = 1

    # Filtrar registros sin ninguna superclase
    mask = df[list(SUPERCLASSES.keys())].sum(axis=1) > 0
    df   = df[mask]

    print(f"  Registros con superclase asignada: {len(df)}")
    print("  Distribución:")
    for sc in SUPERCLASSES:
        n = df[sc].sum()
        print(f"    {sc}: {int(n)} ({100*n/len(df):.1f}%)")

    # Incluir strat_fold para splits reproducibles
    return df[['filename_lr', 'strat_fold'] + list(SUPERCLASSES.keys())]


# ─────────────────────────────────────────────────────────────────────────────
# 2. EXTRAER V1, V2, V3
# ─────────────────────────────────────────────────────────────────────────────
def extract_leads(df: pd.DataFrame, base_dir: Path, records_dir: Path):
    records_dir.mkdir(parents=True, exist_ok=True)

    total      = len(df)
    procesados = 0
    errores    = 0

    print(f"\n  Extrayendo V1/V2/V3 de {total} registros...")
    print(f"  (proceso reanudable — omite los ya procesados)\n")

    for ecg_id, row in df.iterrows():
        out_path = records_dir / f"{ecg_id}.npy"
        if out_path.exists():
            procesados += 1
            continue

        # filename_lr: "records100/00000/00001_lr"
        rel_path  = row['filename_lr']
        full_path = base_dir / rel_path

        try:
            record     = wfdb.rdrecord(str(full_path))
            lead_names = [n.strip() for n in record.sig_name]

            indices = []
            for lead in TARGET_LEADS:
                if lead not in lead_names:
                    raise ValueError(f"Derivación {lead} no encontrada")
                indices.append(lead_names.index(lead))

            signal = record.p_signal[:, indices].astype(np.float32)  # (1000, 3)

            # Normalización z-score por derivación
            for ch in range(3):
                mu, std = signal[:, ch].mean(), signal[:, ch].std()
                if std > 1e-8:
                    signal[:, ch] = (signal[:, ch] - mu) / std

            signal = np.nan_to_num(signal, nan=0.0)
            np.save(out_path, signal)
            procesados += 1

            if procesados % 2000 == 0:
                print(f"  {procesados}/{total} procesados...")

        except Exception as e:
            errores += 1
            if errores <= 3:
                print(f"  ⚠ Error en {ecg_id}: {e}")

    print(f"\n  ✓ Procesados : {procesados}")
    print(f"  ✗ Errores    : {errores}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. GUARDAR LABELS.CSV
# ─────────────────────────────────────────────────────────────────────────────
def save_labels(df: pd.DataFrame, records_dir: Path, out_path: Path):
    valid_ids = [
        ecg_id for ecg_id in df.index
        if (records_dir / f"{ecg_id}.npy").exists()
    ]
    cols     = ['strat_fold'] + list(SUPERCLASSES.keys())
    df_valid = df.loc[valid_ids, cols].copy()
    df_valid.index.name = 'ecg_id'
    df_valid.to_csv(out_path)

    print(f"\n  ✓ Labels guardados: {out_path}")
    print(f"  Total registros válidos: {len(df_valid)}")

    # Mostrar distribución por fold
    print("\n  Distribución por strat_fold:")
    for fold in sorted(df_valid['strat_fold'].unique()):
        n = (df_valid['strat_fold'] == fold).sum()
        print(f"    Fold {int(fold):2d}: {n} registros")

    return df_valid


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 62)
    print("  FASE A — SCRIPT 1: PROCESANDO PTB-XL")
    print("=" * 62)

    # Verificar que existe la carpeta base
    if not BASE_DIR.exists():
        print(f"\n  [ERROR] No se encuentra: {BASE_DIR}")
        print(f"  Verifica que el dataset está en esa ruta.")
        exit(1)

    print(f"\n  Dataset encontrado: {BASE_DIR}")

    # Paso 1
    print("\n" + "=" * 62)
    print("  PARSEANDO ETIQUETAS")
    print("=" * 62)
    df = parse_labels(BASE_DIR)

    # Paso 2
    print("\n" + "=" * 62)
    print("  EXTRAYENDO DERIVACIONES")
    print("=" * 62)
    extract_leads(df, BASE_DIR, RECORDS_DIR)

    # Paso 3
    labels_path = PROCESSED_DIR / 'labels.csv'
    df_final    = save_labels(df, RECORDS_DIR, labels_path)

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  FASE A — SCRIPT 1 COMPLETADO                            ║
╠══════════════════════════════════════════════════════════╣
║  Registros procesados : {len(df_final):>5}                         ║
║  Shape por registro   : (1000, 3) — V1, V2, V3 @ 100Hz  ║
║  Guardados en         : ptb_xl_data/processed/records/   ║
║  Etiquetas en         : ptb_xl_data/processed/labels.csv ║
║                                                          ║
║  SIGUIENTE: ejecuta fase3A_2_data_generator.py           ║
╚══════════════════════════════════════════════════════════╝
    """)
