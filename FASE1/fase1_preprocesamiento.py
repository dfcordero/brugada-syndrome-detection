import wfdb
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import iirnotch, filtfilt, butter

# --- 1. DESCARGA DEL DATASET ---
db_name = 'brugada-huca' # Nombre del dataset en PhysioNet'
dl_dir = './brugada_data'

# Descarga la base de datos si no existe la carpeta
if not os.path.exists(dl_dir):
    print("Descargando dataset Brugada-HUCA desde PhysioNet...")
    wfdb.dl_database(db_name, dl_dir=dl_dir)
    print("¡Descarga completada!")

# --- 2. LECTURA DE UN REGISTRO ---
# Obtenemos la lista de registros y cargamos el primero para probar
records = wfdb.get_record_list(db_name)
test_record_path = os.path.join(dl_dir, records[0])
record = wfdb.rdrecord(test_record_path)

print(f"Registro cargado: {records[0]}")
print(f"Frecuencia de muestreo: {record.fs} Hz")
print(f"Duración: {record.p_signal.shape[0]} muestras ({record.p_signal.shape[0]/record.fs} segundos)")

# --- 3. FILTRADO Y EXTRACCIÓN DE V1, V2, V3 ---
def clean_ecg(signal, fs):
    # Filtro Notch (50 Hz) para ruido de red eléctrica
    b_notch, a_notch = iirnotch(50.0, 30.0, fs)
    sig_notch = filtfilt(b_notch, a_notch, signal)
    
    # Filtro Pasa-banda (0.5 a 40 Hz) para quitar respiración y ruido de alta frecuencia
    b_band, a_band = butter(3, [0.5, 40.0], btype='bandpass', fs=fs)
    return filtfilt(b_band, a_band, sig_notch)

# Identificamos qué columnas corresponden a V1, V2 y V3
leads_of_interest = ['V1', 'V2', 'V3']
lead_indices = [record.sig_name.index(lead) for lead in leads_of_interest]

# Filtramos solo las señales que nos interesan
cleaned_v_leads = {}
for idx, lead in zip(lead_indices, leads_of_interest):
    raw_signal = record.p_signal[:, idx]
    cleaned_v_leads[lead] = clean_ecg(raw_signal, record.fs)

print(f"\nSe han procesado correctamente las derivaciones: {list(cleaned_v_leads.keys())}")