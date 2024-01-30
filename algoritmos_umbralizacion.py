import cv2
import numpy as np

def otsu_thresholding(image):
    image = np.uint8(image)
    # Umbralización de Otsu
    _, th = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return th

def otsu_multi_thresholding(image):
    image = np.uint8(image)
    # Dividir la imagen en regiones (por ejemplo, 4 regiones)
    rows, cols = image.shape
    div_row, div_col = 2, 2
    height, width = rows // div_row, cols // div_col

    # Aplicar umbralización de Otsu en cada región
    for i in range(div_row):
        for j in range(div_col):
            roi = image[i*height:(i+1)*height, j*width:(j+1)*width]
            _, th = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            image[i*height:(i+1)*height, j*width:(j+1)*width] = th
    
    return image


def otsu_multiple(image, num_classes):
    # Convertir la imagen a escala de grises
    gray_image = image

    # Crear un array para almacenar los histogramas de cada clase
    hist_per_class = np.zeros((num_classes, 256), dtype=np.int32)

    # Calcular el histograma total
    hist_total, _ = np.histogram(gray_image.flatten(), bins=256, range=[0, 256])

    # Dividir la imagen en 'num_classes' clases
    step = 256 // num_classes
    for i in range(num_classes):
        lower_bound = i * step
        upper_bound = (i + 1) * step
        hist_per_class[i, :] = np.histogram(
            gray_image[(gray_image >= lower_bound) & (gray_image < upper_bound)].flatten(),
            bins=256, range=[0, 256]
        )[0]

    # Inicializar variables para el cálculo del umbral
    total_pixels = gray_image.size
    total_mean = np.sum(np.arange(256) * hist_total) / total_pixels
    between_class_variances = np.zeros((num_classes, 256), dtype=np.float64)
    optimal_thresholds = np.zeros(num_classes, dtype=np.uint8)

    # Calcular la varianza entre clases para cada umbral posible
    for i in range(num_classes):
        for t in range(256):
            # Calcular la probabilidad de cada clase
            p1 = np.sum(hist_per_class[i, :t]) / total_pixels
            p2 = np.sum(hist_per_class[i, t:]) / total_pixels

            # Calcular la media ponderada de cada clase
            mean1 = np.sum(np.arange(t) * hist_per_class[i, :t]) / (total_pixels * p1 + 1e-10)
            mean2 = np.sum(np.arange(t, 256) * hist_per_class[i, t:]) / (total_pixels * p2 + 1e-10)

            # Calcular la varianza entre clases
            between_class_variances[i, t] = p1 * p2 * (mean1 - mean2)**2

        # Encontrar el umbral óptimo para la clase i
        optimal_thresholds[i] = np.argmax(between_class_variances[i, :])

    # Aplicar umbralización a la imagen original
    segmented_image = np.zeros_like(gray_image, dtype=np.uint8)
    for i in range(num_classes):
        lower_bound = i * step
        upper_bound = (i + 1) * step
        segmented_image[(gray_image >= lower_bound) & (gray_image < upper_bound)] = (
            optimal_thresholds[i] + i * step
        )

    return segmented_image



