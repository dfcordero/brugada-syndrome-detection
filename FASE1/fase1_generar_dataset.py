import wfdb
import os
import glob
import numpy as np
import pandas as pd
from scipy.signal import iirnotch, filtfilt, butter

# Configuración
fs = 100 # Frecuencia de muestreo
leads_of_interest = ['V1', 'V2', 'V3']

def clean_ecg(signal, fs):
    b_notch, a_notch = iirnotch(50.0, 30.0, fs)
    sig_notch = filtfilt(b_notch, a_notch, signal)
    b_band, a_band = butter(3, [0.5, 40.0], btype='bandpass', fs=fs)
    return filtfilt(b_band, a_band, sig_notch)

# Buscar todos los archivos de encabezado (.hea) de forma RECURSIVA
dl_dir = '.' 

record_files = glob.glob(os.path.join(dl_dir, '**', '*.hea'), recursive=True)
record_names = [filepath.replace('.hea', '') for filepath in record_files]

print(f"Archivos .hea encontrados: {len(record_files)}")

# --- NUEVO: Cargar Metadatos Clínicos con Pandas ---
print("Buscando archivo metadata.csv...")
csv_files = glob.glob(os.path.join(dl_dir, '**', 'metadata.csv'), recursive=True)

if len(csv_files) > 0:
    df_meta = pd.read_csv(csv_files[0])
    print("¡Metadatos locales cargados!")
else:
    print("No se encontró localmente. Descargando metadata de PhysioNet...")
    url_csv = "https://physionet.org/files/brugada-huca/1.0.0/metadata.csv"
    df_meta = pd.read_csv(url_csv)

# Asegurar que el patient_id sea texto para cruzarlo sin problemas
df_meta['patient_id'] = df_meta['patient_id'].astype(str)
# ----------------------------------------------------

X = []
y = []

print(f"Procesando {len(record_names)} registros...")

for path in record_names:
    # 1. Extraer etiqueta cruzando el ID del archivo con Pandas
    patient_id = os.path.basename(path) # Extrae el número del archivo (ej: '188981')
    etiqueta_row = df_meta[df_meta['patient_id'] == patient_id]
    
    if len(etiqueta_row) == 0:
        print(f"Saltando {patient_id}: No encontrado en metadatos.")
        continue 

    # Extraemos el valor original (puede ser 0, 1, 2, 3...)
    raw_label = int(etiqueta_row['brugada'].values[0])
    
    # OPTIMIZACIÓN BINARIA: Todo lo que no sea 0 es Brugada (Clase 1)
    label = 1 if raw_label > 0 else 0
    
    # 2. Leer registro y extraer/filtrar señales
    record = wfdb.rdrecord(path)
    lead_indices = [record.sig_name.index(lead) for lead in leads_of_interest]
    
    # Matriz temporal para este paciente (1200 muestras, 3 derivaciones)
    paciente_signals = np.zeros((record.p_signal.shape[0], len(leads_of_interest)))
    
    for i, idx in enumerate(lead_indices):
        raw_signal = record.p_signal[:, idx]
        paciente_signals[:, i] = clean_ecg(raw_signal, fs)
        
    X.append(paciente_signals)
    y.append(label)

# Convertir a arrays de NumPy
X = np.array(X)
y = np.array(y)

# Guardar los arrays para la Fase 2 y 3
np.save('X_data.npy', X)
np.save('y_labels.npy', y)

print("\n--- RESUMEN DEL DATASET ---")
print(f"Dimensión de X (Señales): {X.shape} -> (Pacientes, Muestras, Derivaciones)")
print(f"Dimensión de y (Etiquetas): {y.shape}")
print(f"Casos Brugada (Clase 1): {np.sum(y == 1)}")
print(f"Casos Control (Clase 0): {np.sum(y == 0)}")