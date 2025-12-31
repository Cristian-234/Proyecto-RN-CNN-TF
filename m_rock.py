# =============================================
# CÓDIGO COMPLETO PARA EVALUAR bulbasaur.h5
# Y GENERAR TODAS LAS GRÁFICAS DE EVALUACIÓN
# =============================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report,
    precision_recall_curve, average_precision_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ==================== FUNCIONES DE VISUALIZACIÓN ====================

def plot_roc_curves_multiclass(y_true, y_pred_proba, classes, title="Curvas ROC Multiclasse"):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Micro-average
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    # Macro-average
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    plt.figure(figsize=(14, 10))
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Azar (AUC = 0.50)')

    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    for i, color in zip(range(min(n_classes, 10)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=1.5,
                 label=f'{classes[i]} (AUC = {roc_auc[i]:.3f})')

    plt.plot(fpr["micro"], tpr["micro"],
             label=f'Micro-promedio (AUC = {roc_auc["micro"]:.3f})',
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-promedio (AUC = {roc_auc["macro"]:.3f})',
             color='navy', linestyle=':', linewidth=4)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Falsos Positivos (FPR)')
    plt.ylabel('Verdaderos Positivos (TPR)')
    plt.title(title)
    plt.legend(loc="lower right", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('roc_curves_plantas.png', dpi=300, bbox_inches='tight')
    plt.show()

    return roc_auc


def plot_confusion_matrix_absolute(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Número de muestras'},
                linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicha')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Valores Absolutos')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_absolute_plantas.png', dpi=300, bbox_inches='tight')
    plt.show()
    return cm


def plot_confusion_matrix_normalized(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlGn',
                xticklabels=classes, yticklabels=classes,
                vmin=0, vmax=1, linewidths=0.5, linecolor='gray')
    plt.xlabel('Predicha')
    plt.ylabel('Real')
    plt.title('Matriz de Confusión - Normalizada por Clase Real')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix_normalized_plantas.png', dpi=300, bbox_inches='tight')
    plt.show()
    return cm_norm


def plot_precision_recall_curves(y_true, y_pred_proba, classes):
    n_classes = len(classes)
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    plt.figure(figsize=(14, 10))
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])

    for i, color in zip(range(min(n_classes, 10)), colors):
        precision, recall, _ = precision_recall_curve(y_true_bin[:, i], y_pred_proba[:, i])
        ap = average_precision_score(y_true_bin[:, i], y_pred_proba[:, i])
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{classes[i]} (AP = {ap:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curvas Precision-Recall (Top 10 clases)')
    plt.legend(loc="best", ncol=2)
    plt.grid(True, alpha=0.3)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.tight_layout()
    plt.savefig('precision_recall_curves_plantas.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_class_performance_comparison(y_true, y_pred, classes):
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    f1_scores = [(cls, report[cls]['f1-score']) for cls in classes if cls in report]
    f1_scores.sort(key=lambda x: x[1])
    worst_15 = f1_scores[:15]
    best_15 = f1_scores[-15:]

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    # Peores
    df_worst = pd.DataFrame({
        'precision': [report[c]['precision'] for c, _ in worst_15],
        'recall': [report[c]['recall'] for c, _ in worst_15],
        'f1-score': [report[c]['f1-score'] for c, _ in worst_15]
    }, index=[c for c, _ in worst_15])
    df_worst.plot(kind='barh', ax=axes[0], color=['#d62728', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('15 Clases con Peor Rendimiento')
    axes[0].set_xlim([0, 1.05])

    # Mejores
    df_best = pd.DataFrame({
        'precision': [report[c]['precision'] for c, _ in best_15],
        'recall': [report[c]['recall'] for c, _ in best_15],
        'f1-score': [report[c]['f1-score'] for c, _ in best_15]
    }, index=[c for c, _ in best_15])
    df_best.plot(kind='barh', ax=axes[1], color=['#2ca02c', '#1f77b4', '#9467bd'])
    axes[1].set_title('15 Clases con Mejor Rendimiento')
    axes[1].set_xlim([0, 1.05])

    plt.tight_layout()
    plt.savefig('class_performance_comparison_plantas.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_auc_scores_distribution(roc_auc, classes):
    auc_scores = sorted([(classes[i], roc_auc[i]) for i in range(len(classes))], key=lambda x: x[1])
    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    worst = auc_scores[:15]
    best = auc_scores[-15:]
    axes[0].barh([x[0] for x in worst], [x[1] for x in worst], color='salmon')
    axes[0].set_title('15 Clases con Menor AUC')
    axes[1].barh([x[0] for x in best], [x[1] for x in best], color='lightgreen')
    axes[1].set_title('15 Clases con Mayor AUC')
    for ax in axes:
        ax.set_xlim([0, 1.05])
        ax.axvline(0.5, color='red', linestyle='--')
    plt.tight_layout()
    plt.savefig('auc_distribution_plantas.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_prediction_distribution(y_true, y_pred, classes):
    true_counts = np.bincount(y_true, minlength=len(classes))
    pred_counts = np.bincount(y_pred, minlength=len(classes))
    freq = sorted(enumerate(true_counts), key=lambda x: x[1], reverse=True)[:20]
    indices = [i for i, _ in freq]
    labels = [classes[i] for i in indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    axes[0].bar(range(20), [true_counts[i] for i in indices], color='skyblue')
    axes[0].set_xticks(range(20))
    axes[0].set_xticklabels(labels, rotation=45, ha='right')
    axes[0].set_title('Top 20 Clases Reales')
    axes[1].bar(range(20), [pred_counts[i] for i in indices], color='lightcoral')
    axes[1].set_xticks(range(20))
    axes[1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1].set_title('Top 20 Predicciones')
    plt.tight_layout()
    plt.savefig('prediction_distribution_plantas.png', dpi=300, bbox_inches='tight')
    plt.show()


def generate_detailed_statistics(y_true, y_pred, classes):
    accuracy = np.mean(y_true == y_pred)
    report = classification_report(y_true, y_pred, target_names=classes, output_dict=True, zero_division=0)
    
    print("="*80)
    print("ESTADÍSTICAS DETALLADAS DEL MODELO")
    print("="*80)
    print(f"Accuracy Global: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Muestras: {len(y_true)} | Clases: {len(classes)}")
    print(f"\nMacro Avg  → Precision: {report['macro avg']['precision']:.4f} | "
          f"Recall: {report['macro avg']['recall']:.4f} | F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Weighted Avg → Precision: {report['weighted avg']['precision']:.4f} | "
          f"Recall: {report['weighted avg']['recall']:.4f} | F1: {report['weighted avg']['f1-score']:.4f}")

    # Top 10 mejores y peores F1
    f1_list = [(c, report[c]['f1-score'], report[c]['support']) for c in classes if c in report]
    f1_list.sort(key=lambda x: x[1], reverse=True)
    print("\nTOP 10 MEJORES F1-SCORE")
    for i, (c, f1, s) in enumerate(f1_list[:10], 1):
        print(f"{i:2}. {c:40} F1: {f1:.4f} (samples: {int(s)})")

    print("\nTOP 10 PEORES F1-SCORE")
    for i, (c, f1, s) in enumerate(reversed(f1_list[-10:]), 1):
        print(f"{i:2}. {c:40} F1: {f1:.4f} (samples: {int(s)})")


# ==================== FUNCIÓN PRINCIPAL ====================

def evaluar_modelo_bulbasaur():
    # ----------------- AJUSTA ESTAS RUTAS -----------------
    modelo_path = 'modelo/bulbasaur.h5'                    # Ruta a tu modelo
    validacion_dir = 'data'  # Carpeta con subcarpetas por clase

    img_height, img_width = 224, 224  # Ajusta si usaste otro tamaño (ej. 224 para MobileNetV2)
    batch_size = 32

    # ----------------- CARGA DEL MODELO -----------------
    print("Cargando modelo bulbasaur.h5...")
    model = load_model(modelo_path)

    # ----------------- GENERADOR DE VALIDACIÓN -----------------
    val_datagen = ImageDataGenerator(rescale=1./255)  # Si usaste preprocess_input de MobileNet, cámbialo

    val_gen = val_datagen.flow_from_directory(
        validacion_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False  # ¡Importante!
    )

    categories_aug = list(val_gen.class_indices.keys())
    print(f"\nClases detectadas ({len(categories_aug)}): {categories_aug}")

    # ----------------- PREDICCIONES -----------------
    print("\nGenerando predicciones...")
    y_pred_proba = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = val_gen.classes

    # ----------------- GENERAR TODAS LAS GRÁFICAS -----------------
    print("\nGenerando gráficas...")

    roc_auc = plot_roc_curves_multiclass(y_true, y_pred_proba, categories_aug)
    plot_confusion_matrix_absolute(y_true, y_pred, categories_aug)
    plot_confusion_matrix_normalized(y_true, y_pred, categories_aug)
    plot_precision_recall_curves(y_true, y_pred_proba, categories_aug)
    plot_class_performance_comparison(y_true, y_pred, categories_aug)
    plot_auc_scores_distribution(roc_auc, categories_aug)
    plot_prediction_distribution(y_true, y_pred, categories_aug)
    generate_detailed_statistics(y_true, y_pred, categories_aug)

    print("\n¡Evaluación completa finalizada!")
    print("Archivos generados en el directorio actual:")
    print("  • roc_curves_plantas.png")
    print("  • confusion_matrix_absolute_plantas.png")
    print("  • confusion_matrix_normalized_plantas.png")
    print("  • precision_recall_curves_plantas.png")
    print("  • class_performance_comparison_plantas.png")
    print("  • auc_distribution_plantas.png")
    print("  • prediction_distribution_plantas.png")


# ==================== EJECUTAR ====================
if __name__ == "__main__":
    evaluar_modelo_bulbasaur()