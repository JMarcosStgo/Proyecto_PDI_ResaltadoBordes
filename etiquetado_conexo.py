import numpy as np
import cv2
from skimage import measure

def etiquetado_componentes_conexos(imagen_gris):
    # Binarizar la imagen
    binary_image = imagen_gris #"imagen_gris > umbral

    # Etiquetar los componentes conexos
    labeled_image, num_labels = measure.label(binary_image, connectivity=2, return_num=True)

    return labeled_image, num_labels

def etiquetado_componentes_conexos2(imagen_binaria):
    def vecindad(y, x, image_shape):
        vecinos = []
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < image_shape[0] and 0 <= nx < image_shape[1]:
                    vecinos.append((ny, nx))
        return vecinos

    def dfs(y, x, label, image, visited, labels):
        visited[y][x] = True
        labels[y][x] = label
        for ny, nx in vecindad(y, x, image.shape):
            if image[ny][nx] == 1 and not visited[ny][nx]:
                dfs(ny, nx, label, image, visited, labels)

    image_shape = imagen_binaria.shape
    visited = [[False for _ in range(image_shape[1])] for _ in range(image_shape[0])]
    labels = [[0 for _ in range(image_shape[1])] for _ in range(image_shape[0])]
    label_count = 0

    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            if imagen_binaria[y][x] == 1 and not visited[y][x]:
                label_count += 1
                dfs(y, x, label_count, imagen_binaria, visited, labels)

    return labels, label_count

def escala_de_grises_a_binaria(imagen_gris, umbral):
    # Binarizar la imagen
    imagen_binaria =  imagen_gris > umbral #cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    return imagen_binaria

def etiquetado_componentes_conexos_large(imagen_binaria):
    # Etiquetar los componentes conexos
    labeled_image, num_labels = etiquetado_componentes_conexos(imagen_binaria) #measure.label(imagen_binaria, connectivity=2, return_num=True)
    
    # Calcular el tamaño de cada región etiquetada
    region_sizes = [np.sum(labeled_image == label) for label in range(1, num_labels + 1)]
    
    # Encontrar la etiqueta de la región más grande
    max_region_label = np.argmax(region_sizes) + 1  # Sumamos 1 porque las etiquetas comienzan desde 1
    
    # Obtener únicamente la región más grande
    largest_region = np.where(labeled_image == max_region_label, 1, 0)
    
    return largest_region

def obtener_region_hoja(imagen_binaria):
    # Etiquetar los componentes conexos
    labeled_image, num_labels = measure.label(imagen_binaria, connectivity=2, return_num=True)
    
    # Calcular el tamaño de cada región etiquetada
    region_sizes = [np.sum(labeled_image == label) for label in range(1, num_labels + 1)]
    
    # Encontrar la etiqueta de la región más grande
    max_region_label = np.argmax(region_sizes) + 1  # Sumamos 1 porque las etiquetas comienzan desde 1
    
    # Obtener únicamente la región más grande (la hoja)
    hoja = np.where(labeled_image == max_region_label, 1, 0)
    
    return hoja

def recuperar_valores_originales(imagen_etiquetada, imagen_gris):
    # Crear una copia de la imagen en escala de grises original
    imagen_recuperada = np.copy(imagen_gris)

    # Recuperar los valores originales de intensidad para los píxeles igual a 1 (region de la hoja)
    imagen_recuperada[imagen_etiquetada == 1] = imagen_gris[imagen_etiquetada == 1]

    return imagen_recuperada

def recuperar_valores_originales2(imagen_etiquetada, imagen_gris):
    # Crear una copia de la imagen en escala de grises original
    imagen_recuperada = np.zeros_like(imagen_gris)

    # Recuperar los valores originales de intensidad para los píxeles igual a 1 (region de la hoja)
    indices_unos = imagen_etiquetada == 1
    imagen_recuperada[indices_unos] = imagen_gris[indices_unos]

    return imagen_recuperada