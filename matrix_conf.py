import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ==================== CONFIGURACIÓN ====================
MODEL_PATH = 'modelo/bulbasaur.h5'          # Ruta a tu modelo
DATA_DIR = 'data'                           # Carpeta con subcarpetas por clase (sin augmentation)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# =======================================================

# 1. Generador de validación
val_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input
    # Cambia a mobilenet.preprocess_input si usaste MobileNetV1
)

val_generator = val_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # ¡Importante para alinear predicciones!
)

# 2. Cargar modelo
print("Cargando el modelo...")
model = tf.keras.models.load_model(MODEL_PATH)

# 3. Predicciones
print("Generando predicciones...")
val_generator.reset()
y_pred_prob = model.predict(val_generator, verbose=1)
y_pred = np.argmax(y_pred_prob, axis=1)
y_true = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# ==================== ABREVIAR ETIQUETAS ====================
def abbreviate_name(name):
    # El dataset usa formato: Planta___enfermedad o Planta___healthy
    if '___' in name:
        plant, condition = name.split('___')
        # Mapa de nombres de plantas más cortos
        plant_map = {
            'Tomato': 'Tomato',
            'Apple': 'Apple',
            'Corn': 'Maíz',
            'Potato': 'Papa',
            'Grape': 'Uva',
            'Cherry': 'Cereza',
            'Peach': 'Durazno',
            'Pepper': 'Pimiento',
            'Strawberry': 'Fresa',
            'Blueberry': 'Arándano',
            'Raspberry': 'Frambuesa',
            'Soybean': 'Soja',
            'Squash': 'Calabaza',
            'Orange': 'Naranja'
        }
        plant_short = plant_map.get(plant, plant)
        
        if condition == 'healthy':
            return f"{plant_short} (Sano)"
        else:
            # Abreviar enfermedades largas
            short_cond = condition.replace('_', ' ').replace(' blight', '').replace(' rot', '')
            short_cond = short_cond.replace(' bacterial spot', ' bact.')
            short_cond = short_cond.replace(' early', ' temp.')
            short_cond = short_cond.replace(' late', ' tard.')
            short_cond = short_cond.replace(' leaf', '')
            short_cond = short_cond.replace(' powdery mildew', ' oídio')
            return f"{plant_short} ({short_cond[:12]})"
    else:
        return name[:18]  # Para Background_without_leaves u otros

short_labels = [abbreviate_name(name) for name in class_names]

# ==================== MATRICES DE CONFUSIÓN ====================
cm = confusion_matrix(y_true, y_pred)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # Normalizada por fila

# Configuración estética
sns.set(font_scale=1.0)
fig, axes = plt.subplots(1, 2, figsize=(32, 14), gridspec_kw={'width_ratios': [1, 1]})

# 1. Matriz absoluta
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.5, linecolor='lightgray', cbar_kws={"shrink": 0.8})
axes[0].set_title('Matriz de Confusión\n(Valores Absolutos)', fontsize=16, pad=20)
axes[0].set_xlabel('Predicción', fontsize=12)
axes[0].set_ylabel('Real', fontsize=12)

# 2. Matriz normalizada (porcentajes)
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens', ax=axes[1],
            xticklabels=short_labels, yticklabels=short_labels,
            linewidths=0.5, linecolor='lightgray', cbar_kws={"shrink": 0.8})
axes[1].set_title('Matriz de Confusión\n(Normalizada por Clase Real)', fontsize=16, pad=20)
axes[1].set_xlabel('Predicción', fontsize=12)
axes[1].set_ylabel('Real', fontsize=12)

# Rotar etiquetas para mejor legibilidad
for ax in axes:
    ax.tick_params(axis='x', rotation=90, labelsize=10)
    ax.tick_params(axis='y', rotation=0, labelsize=10)

plt.suptitle('Análisis del Modelo Bulbasaur - 38 Clases de Enfermedades en Plantas', 
             fontsize=20, y=1.02)

plt.tight_layout()

# ==================== GUARDAR IMÁGENES ====================
# Crear carpeta de salida si no existe
os.makedirs('resultados', exist_ok=True)

# Guardar la figura combinada
combined_path = 'resultados/matriz_confusion_comparacion.png'
plt.savefig(combined_path, dpi=300, bbox_inches='tight')
print(f"\nFigura combinada guardada en: {combined_path}")

# Opcional: guardar por separado
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=short_labels, yticklabels=short_labels)
plt.title('Matriz de Confusión - Valores Absolutos')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('resultados/matriz_confusion_absoluta.png', dpi=300, bbox_inches='tight')

plt.figure(figsize=(16, 14))
sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=short_labels, yticklabels=short_labels)
plt.title('Matriz de Confusión - Porcentajes')
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('resultados/matriz_confusion_normalizada.png', dpi=300, bbox_inches='tight')

print("Matriz absoluta guardada en: resultados/matriz_confusion_absoluta.png")
print("Matriz normalizada guardada en: resultados/matriz_confusion_normalizada.png")

# Mostrar en pantalla (opcional)
plt.show()

# ==================== REPORTE DETALLADO ====================
print("\n" + "="*80)
print("REPORTE DE CLASIFICACIÓN DETALLADO")
print("="*80)
print(classification_report(y_true, y_pred, target_names=short_labels, digits=4))