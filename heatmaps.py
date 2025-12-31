import tensorflow as tf
import numpy as np
import cv2


# ======================================================
# 1. MAPA DE CALOR BASADO EN GRADIENTES (Interpretabilidad)
# ======================================================
def make_gradcam_heatmap(img_array, model, pred_index=None):
    """
    Genera un mapa de calor basado en gradientes.
    Funciona con modelos encapsulados (MobileNetV2 + Sequential).
    
    img_array: numpy array (1, 224, 224, 3)
    model: modelo Keras cargado
    pred_index: clase objetivo (opcional)
    """

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(img_tensor)
        predictions = model(img_tensor)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])

        loss = predictions[:, pred_index]

    # Gradientes respecto a la imagen de entrada
    grads = tape.gradient(loss, img_tensor)

    # Promedio del gradiente por canal
    heatmap = tf.reduce_mean(tf.abs(grads), axis=-1)

    heatmap = heatmap[0]
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ======================================================
# 2. MAPA DE VERIFICACIÓN DE DECISIONES
# ======================================================
def activation_heatmap(img_array, model):
    """
    Genera un mapa de activación simple para verificar
    la respuesta global del modelo.
    """

    preds = model(img_array)

    heatmap = tf.reduce_mean(preds, axis=-1)
    heatmap = tf.maximum(heatmap, 0)
    heatmap /= tf.reduce_max(heatmap) + 1e-8

    return heatmap.numpy()


# ======================================================
# 3. SUPERPOSICIÓN DEL MAPA DE CALOR
# ======================================================
def superimpose_heatmap(heatmap, image, alpha=0.4):
    """
    Superpone el mapa de calor sobre la imagen original.

    heatmap: numpy array (224, 224)
    image: imagen original (H, W, 3)
    alpha: transparencia
    """

    heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(
        image, 1 - alpha, heatmap, alpha, 0
    )

    return superimposed_img
