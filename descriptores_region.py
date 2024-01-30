import cv2
import numpy as np
from tabulate import tabulate


def calculate_compactness_and_hu_moments(image):
    # Cargar la imagen
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Buscar contornos en la imagen
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcular el área y el perímetro del objeto más grande
    largest_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Calcular la compacidad
    compactness = (perimeter ** 2) / (4 * np.pi * area)

    # Calcular los momentos de Hu
    hu_moments = cv2.HuMoments(cv2.moments(largest_contour)).flatten()

    return compactness, hu_moments


def calculate_indices_and_texture_descriptors(image):
    # Cargar la imagen
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Buscar contornos en la imagen
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calcular el centroide de la imagen
    M = cv2.moments(image)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
    centroid = (centroid_x, centroid_y)

    # Calcular las distancias radiales desde el centroide a los puntos del contorno
    distances = [np.linalg.norm(np.array([centroid_x, centroid_y]) - contour.squeeze(), axis=1) for contour in contours]

    # Calcular el índice de área-distancia radial normalizada
    normalized_area_distance_index = np.sum([np.sum(distances[i]) / cv2.contourArea(contours[i]) for i in range(len(contours))])

    # Calcular la longitud de los contornos
    contour_lengths = [cv2.arcLength(contour, True) for contour in contours]

    # Calcular el índice de rugosidad-distancia radial normalizada
    normalized_roughness_index = np.sum([contour_lengths[i] / np.max(distances[i]) for i in range(len(contours))])

    # Calcular la entropía y la uniformidad
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    hist_normalized = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
    uniformity = np.sum(hist_normalized ** 2)

    # Calcular los descriptores de textura de primer orden
    mean_intensity = np.mean(image)
    std_dev_intensity = np.std(image)
    skewness_intensity = np.mean(((image - mean_intensity) / std_dev_intensity) ** 3)
    kurtosis_intensity = np.mean(((image - mean_intensity) / std_dev_intensity) ** 4) - 3

    return normalized_area_distance_index, normalized_roughness_index, entropy, uniformity, mean_intensity, std_dev_intensity, skewness_intensity, kurtosis_intensity


def print_results_compactness_hu(compactness, hu_moments):
    print("Resultados de Compacidad y Momentos Hu:")
    compactness_row = ["Compacidad", compactness]
    hu_moments_row = ["Momentos Hu"] + hu_moments.tolist()
    print(tabulate([compactness_row, hu_moments_row], headers=["Descriptor", "Valor"], tablefmt="fancy_grid"))
    print("\n")

def print_results_texture_descriptors(indices_and_descriptors):
    print("Resultados de Índices y Descriptores de Textura:")
    headers = ["Descriptor", "Valor"]
    rows = [
        ["Índice área-distancia radial normalizada", indices_and_descriptors[0]],
        ["Índice rugosidad-distancia radial normalizada", indices_and_descriptors[1]],
        ["Entropía", indices_and_descriptors[2]],
        ["Uniformidad", indices_and_descriptors[3]],
        ["Media de intensidad", indices_and_descriptors[4]],
        ["Desviación estándar de intensidad", indices_and_descriptors[5]]
    ]
    rows = [list(map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x, row)) for row in rows]
    print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))
    print("\n")