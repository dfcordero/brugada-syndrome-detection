"""
================================================================================
  GRAD-CAM — Explicabilidad de predicciones ECG
================================================================================
  Ejecutar desde: ~/proyectos/brugada_syndrom/FASE3_PTB/
  python fase3C_gradcam.py

  Grad-CAM (Gradient-weighted Class Activation Mapping):
    Calcula qué regiones temporales de la señal ECG activaron más
    la última capa convolucional al hacer la predicción.
    → Picos altos = el modelo "miró" ahí para decidir
    → Clínicamente válido si coincide con el segmento ST en V1/V2

  Visualiza 4 tipos de casos por modelo:
    · TP Brugada — clasificados correctamente
    · FP Control — sanos clasificados como Brugada (¿qué confunde al modelo?)
    · TN Control — clasificados correctamente
    · FN Brugada — Brugada no detectados (¿qué falla?)
================================================================================
"""

from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)

BRUGADA_DIR = Path('../FASE2')
LEADS       = ['V1', 'V2', 'V3']
FS          = 100  # Hz
T_AXIS      = np.arange(1000) / FS  # eje temporal en segundos


# ─────────────────────────────────────────────────────────────────────────────
# 1. CARGAR DATOS
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 62)
print("  CARGANDO DATOS")
print("=" * 62)

X_test  = np.load(BRUGADA_DIR / 'X_test.npy')
y_test  = np.load(BRUGADA_DIR / 'y_test.npy')
X_test_1000 = X_test[:, :1000, :]

feat_mean = np.load(BRUGADA_DIR / 'feat_mean.npy')
feat_std  = np.load(BRUGADA_DIR / 'feat_std.npy')

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
            feats[i, base+4] = fft_m[(freqs>=0.5)&(freqs<=5)].sum() / e
            feats[i, base+5] = fft_m[(freqs>=5) &(freqs<=40)].sum() / e
    return feats

F_test = (extract_features(X_test) - feat_mean) / feat_std
print(f"  Test: {X_test.shape[0]} pacientes")


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRAD-CAM PARA CNN 1D
# ─────────────────────────────────────────────────────────────────────────────

def compute_gradcam_1d(model, X_ecg, X_feat, conv_layer_name='res2_r2'):
    """
    Calcula Grad-CAM para una señal ECG 1D.

    Pasos:
      1. Forward pass guardando activaciones de la última Conv (res2_r2)
      2. Calcula gradiente de la predicción respecto a esas activaciones
      3. Pondera los mapas de activación por sus gradientes (importancia)
      4. Aplica ReLU y normaliza a [0,1]
      5. Interpola al tamaño original de la señal

    Args:
        model         : modelo Keras
        X_ecg         : señal (1, T, 3)
        X_feat        : features estadísticos (1, 18)
        conv_layer_name: nombre de la capa convolucional objetivo

    Returns:
        cam : array (T,) con importancia temporal normalizada [0,1]
    """
    # Modelo que devuelve activaciones de la capa conv + predicción final
    grad_model = tf.keras.Model(
        inputs  = model.inputs,
        outputs = [model.get_layer(conv_layer_name).output, model.output]
    )

    X_ecg_tf  = tf.cast(tf.expand_dims(X_ecg[0], 0), tf.float32)
    X_feat_tf = tf.cast(tf.expand_dims(X_feat[0], 0), tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(
            [X_ecg_tf, X_feat_tf], training=False
        )
        pred_class = tf.cast(predictions[0, 0], tf.float32)

    # Gradiente de la predicción respecto a los mapas de activación
    grads = tape.gradient(pred_class, conv_outputs)  # (1, T', filters)

    # Pooling global de gradientes por filtro → importancia de cada filtro
    pooled_grads = tf.reduce_mean(grads, axis=[0, 1])  # (filters,)

    # Ponderar activaciones por importancia de cada filtro
    conv_outputs = conv_outputs[0]  # (T', filters)
    cam = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)  # (T',)

    # ReLU — solo importan las activaciones positivas
    cam = tf.nn.relu(cam).numpy()

    # Interpolar al tamaño original de la señal
    if cam.max() > 0:
        cam = cam / cam.max()

    # Upsampling lineal a longitud de entrada
    T_orig = X_ecg.shape[1]
    T_cam  = len(cam)
    cam_interp = np.interp(
        np.linspace(0, T_cam-1, T_orig),
        np.arange(T_cam),
        cam
    )
    return cam_interp


# ─────────────────────────────────────────────────────────────────────────────
# 3. FUNCIÓN DE VISUALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def plot_gradcam_cases(model, X_ecg_all, X_feat_all, y_true, y_prob,
                       threshold, model_name, signal_len=1000,
                       conv_layer='res2_r2', save_path=None):
    """
    Genera figura con Grad-CAM para 4 tipos de casos:
    TP, FP, TN, FN — 3 derivaciones por caso.
    """
    y_pred = (y_prob > threshold).astype(int)

    # Seleccionar índices de cada tipo
    tp_idx = np.where((y_true == 1) & (y_pred == 1))[0]
    fp_idx = np.where((y_true == 0) & (y_pred == 1))[0]
    tn_idx = np.where((y_true == 0) & (y_pred == 0))[0]
    fn_idx = np.where((y_true == 1) & (y_pred == 0))[0]

    cases = [
        (tp_idx, 'TP — Brugada detectado ✓',    '#C62828', 'Brugada'),
        (fp_idx, 'FP — Sano clasificado Brugada ✗', '#FF8F00', 'Control'),
        (tn_idx, 'TN — Sano detectado ✓',        '#1565C0', 'Control'),
        (fn_idx, 'FN — Brugada no detectado ✗',  '#6A1B9A', 'Brugada'),
    ]

    fig = plt.figure(figsize=(20, 14))
    fig.suptitle(f'Grad-CAM — {model_name}\n'
                 f'Zonas rojas = donde miró el modelo para decidir',
                 fontsize=13, fontweight='bold')

    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.3)

    for row, (indices, case_title, color, true_label) in enumerate(cases):
        if len(indices) == 0:
            for col in range(3):
                ax = fig.add_subplot(gs[row, col])
                ax.text(0.5, 0.5, 'Sin casos\nen test set',
                        ha='center', va='center', transform=ax.transAxes,
                        fontsize=10, color='gray')
                ax.set_title(f'{case_title}\n{LEADS[col]}', fontsize=9)
                ax.axis('off')
            continue

        # Tomar el primer caso disponible
        idx = indices[0]
        X_ecg  = X_ecg_all[idx:idx+1]   # (1, T, 3)
        X_feat = X_feat_all[idx:idx+1]   # (1, 18)
        prob   = y_prob[idx]

        # Calcular Grad-CAM
        try:
            cam = compute_gradcam_1d(model, X_ecg, X_feat,
                                     conv_layer_name=conv_layer)
        except Exception as e:
            print(f"  ⚠ Error Grad-CAM en {case_title}: {e}")
            cam = np.zeros(signal_len)

        # Plot por derivación
        for col, lead in enumerate(LEADS):
            ax = fig.add_subplot(gs[row, col])
            signal = X_ecg[0, :, col]

            # Señal ECG
#FIX
            t_axis_real = np.arange(len(signal)) / FS
            ax.plot(t_axis_real, signal, color='#333333', lw=0.8, alpha=0.9,
                    label='ECG')

            # Heatmap Grad-CAM como área sombreada
            ax.fill_between(t_axis_real, signal.min(), signal.max(),
                            alpha=cam * 0.45, color='#E24B4A')

            # Línea de importancia normalizada (eje secundario)
            ax2 = ax.twinx()
            ax2.plot(t_axis_real, cam, color='#E24B4A', lw=1.2, alpha=0.7,
                     ls='--', label='Grad-CAM')
            ax2.set_ylim(0, 2.5)
            ax2.set_yticks([])

            ax.set_title(f'{case_title}\n{lead} | P(Brugada)={prob:.3f}',
                         fontsize=8, fontweight='bold', color=color)
            ax.set_xlabel('Tiempo (s)', fontsize=7)
            ax.set_ylabel('Amplitud (norm.)', fontsize=7)
            ax.tick_params(labelsize=7)
            ax.grid(alpha=0.2)

            # Marcar zona ST (aprox 0.06-0.16s después del inicio del QRS)
            # En un ECG de 10s a 100Hz hay ~8-10 latidos
            # Marcamos aproximaciones de dónde debería estar el ST
            ax.axvspan(0.3, 0.5, alpha=0.08, color='green',
                       label='Zona QRS aprox')

    # Leyenda global
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0],[0], color='#333333', lw=1.5, label='Señal ECG'),
        Patch(facecolor='#E24B4A', alpha=0.5, label='Grad-CAM (zona atendida)'),
        Patch(facecolor='green',   alpha=0.1, label='Zona QRS aprox'),
    ]
    fig.legend(handles=legend_elements, loc='lower center',
               ncol=3, fontsize=9, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Guardado: {save_path}")
    plt.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. CARGAR MODELOS Y EJECUTAR
# ─────────────────────────────────────────────────────────────────────────────

def bce_label_smoothing(epsilon=0.05):
    def loss_fn(y_true, y_pred):
        y_true   = tf.cast(y_true, tf.float32)
        y_pred   = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0-1e-7)
        y_smooth = y_true*(1.0-epsilon)+epsilon/2.0
        return tf.reduce_mean(
            -(y_smooth*tf.math.log(y_pred) +
              (1-y_smooth)*tf.math.log(1-y_pred))
        )
    loss_fn.__name__ = 'bce_smooth_0.05'
    return loss_fn


# ── v4.1 — LSTM, input (1000, 3) ─────────────────────────────────────────────
print("\n" + "=" * 62)
print("  GRAD-CAM — v4.1 (TL + LSTM)")
print("=" * 62)

model_v41 = tf.keras.models.load_model(
    'modelo_brugada_v41.keras',
    custom_objects={'bce_smooth_0.05': bce_label_smoothing(0.05)}
)

umbral_v41 = float(np.load('umbral_v41.npy')[0])
y_prob_v41 = model_v41.predict([X_test_1000, F_test], verbose=0).flatten()
auc_v41    = roc_auc_score(y_test, y_prob_v41)

print(f"  AUC: {auc_v41:.4f}  |  Umbral: {umbral_v41:.3f}")
cm_v41 = confusion_matrix(y_test, (y_prob_v41 > umbral_v41).astype(int))
print(f"  [[{cm_v41[0,0]:>3} {cm_v41[0,1]:>3}]  TN | FP")
print(f"   [{cm_v41[1,0]:>3} {cm_v41[1,1]:>3}]] FN | TP")

plot_gradcam_cases(
    model      = model_v41,
    X_ecg_all  = X_test_1000,
    X_feat_all = F_test,
    y_true     = y_test,
    y_prob     = y_prob_v41,
    threshold  = umbral_v41,
    model_name = 'v4.1 — Transfer Learning + LSTM (AUC=0.987, FP=2, FN=1)',
    signal_len = 1000,
    conv_layer = 'res2_r2',
    save_path  = 'gradcam_v41.png'
)


# ── v4.2 — GAP, input (1200, 3) ──────────────────────────────────────────────
print("\n" + "=" * 62)
print("  GRAD-CAM — v4.2 (TL + GAP + Dense)")
print("=" * 62)

model_v42 = tf.keras.models.load_model(
    'modelo_brugada_v42.keras',
    custom_objects={'bce_smooth_0.05': bce_label_smoothing(0.05)}
)

umbral_v42 = float(np.load('umbral_v42.npy')[0])
y_prob_v42 = model_v42.predict([X_test, F_test], verbose=0).flatten()
auc_v42    = roc_auc_score(y_test, y_prob_v42)

print(f"  AUC: {auc_v42:.4f}  |  Umbral: {umbral_v42:.3f}")
cm_v42 = confusion_matrix(y_test, (y_prob_v42 > umbral_v42).astype(int))
print(f"  [[{cm_v42[0,0]:>3} {cm_v42[0,1]:>3}]  TN | FP")
print(f"   [{cm_v42[1,0]:>3} {cm_v42[1,1]:>3}]] FN | TP")

plot_gradcam_cases(
    model      = model_v42,
    X_ecg_all  = X_test,
    X_feat_all = F_test,
    y_true     = y_test,
    y_prob     = y_prob_v42,
    threshold  = umbral_v42,
    model_name = 'v4.2 — Transfer Learning + GAP (AUC=0.978, FP=4, FN=0)',
    signal_len = 1000,   # visualizamos los primeros 10s para comparar
    conv_layer = 'res2_r2',
    save_path  = 'gradcam_v42.png'
)

print("""
╔══════════════════════════════════════════════════════════╗
║  GRAD-CAM COMPLETADO                                     ║
╠══════════════════════════════════════════════════════════╣
║  gradcam_v41.png — v4.1 LSTM                            ║
║  gradcam_v42.png — v4.2 GAP+Dense                       ║
║                                                          ║
║  Interpretar:                                            ║
║  · Zona roja intensa = modelo mirando ahí                ║
║  · Si coincide con QRS/ST en V1/V2 = comportamiento     ║
║    clínicamente correcto                                 ║
║  · Si mira al final de la señal o en V3 = artefacto     ║
╚══════════════════════════════════════════════════════════╝
""")
