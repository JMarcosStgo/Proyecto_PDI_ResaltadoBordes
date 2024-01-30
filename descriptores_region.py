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
    compactness =  area / ( perimeter **2 / (4 * np.pi))

    # Calcular los momentos de Hu
    #Los momentos de Hu se calculan a partir de los momentos centrales utilizando la función cv2.HuMoments(). La función flatten() se utiliza para aplanar la matriz de momentos de Hu en un vector unidimensional 1
    hu_moments = cv2.HuMoments(cv2.moments(largest_contour)).flatten()
    print("hu_moments",hu_moments)
    return compactness, hu_moments


def calculate_indices_and_texture_descriptors(image):
    # Cargar la imagen
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)

    # Buscar contornos en la imagen
        #obtenemos una lista de contornos, donde cada contorno es una matriz de puntos que representan las coordenadas de los bordes de la figura en la imagen.
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #print("contours",contours)

    # Calcular el centroide de la imagen
    M = cv2.moments(image)
    centroid_x = int(M["m10"] / M["m00"])
    centroid_y = int(M["m01"] / M["m00"])
   
    # Calcular las distancias euclidianas entre el centroide y cada punto del contorno
    distances = []
    for contour in contours:
        for point in contour.squeeze():
            distance = np.linalg.norm(np.array([centroid_x, centroid_y]) - point)
            distances.append(distance)

    # Convertir la lista de distancias a un vector numpy
    distances_vector = np.array(distances)

    #print("Vector de distancias:",len(distances_vector), distances_vector)

    max_distance = np.max(distances_vector)
    #print("Máximo valor de las distancias:", max_distance)
    distances_normalized = distances_vector / max_distance
    #print("distances_normalized",distances_normalized)

    # Calcular el índice de área
    area_index = np.sum(distances_normalized)
    #print("Índice de área:", area_index)

    # Calcular la media del vector distances_normalized
    mean_distance_normalized = np.mean(distances_normalized)

    # Generar un nuevo vector con los valores ajustados
    index_area = np.where(distances_normalized > mean_distance_normalized, distances_normalized - mean_distance_normalized, 0)
    #print("Nuevo vector ajustado:", index_area)

    #obtenemos el area
    normalized_area_distance_index = (1/ ( len(index_area ) * mean_distance_normalized )) * np.sum(index_area)
    #print("normalized_area_distance_index",normalized_area_distance_index)



    vector_rugosidad = []
    for x in range(len(distances_normalized)-1):
        vector_rugosidad.append(distances_normalized[x]-distances_normalized[x+1])
    sum_vector_rugosidad = np.sum(vector_rugosidad)
    normalized_roughness_index = (1/ ( (len(index_area ) -1) * mean_distance_normalized )) * np.sum(sum_vector_rugosidad)
    #print("normalized_roughness_index",normalized_roughness_index)


    # Calcular los descriptores de textura de primer orden
    #media,varianza y desviacion estandar
    mean_intensity = np.mean(image)
    var = np.var(image)
    std_dev_intensity = 1 - (1/(1 + var)) #np.std(image)
    # Calcular la entropía y la uniformidad
    hist = cv2.calcHist([image], [0], None, [256], [0,256])
    # Probabilidad de p
    hist_normalized = hist.ravel() / hist.sum()
    entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-8))
    # Suma de p^2
    uniformity = np.sum(hist_normalized ** 2)

    
    skewness_intensity = np.mean(((image - mean_intensity) / std_dev_intensity) ** 3)
    kurtosis_intensity = np.mean(((image - mean_intensity) / std_dev_intensity) ** 4) - 3

    return normalized_area_distance_index, normalized_roughness_index, entropy, uniformity, mean_intensity, std_dev_intensity, skewness_intensity, kurtosis_intensity


def print_results_compactness_hu(compactness, hu_moments):
    print("Resultados de Compacidad y Momentos Hu:")
    compactness_row = ["Compacidad", compactness]
    hu_moments_rows = [["Momentos Hu {}".format(i+1), hu_moment] for i, hu_moment in enumerate(hu_moments)]
    print(tabulate([compactness_row] + hu_moments_rows, headers=["Descriptor", "Valor"], tablefmt="fancy_grid"))
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