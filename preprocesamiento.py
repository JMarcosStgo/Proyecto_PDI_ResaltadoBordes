# Importar la librería de opencv
import cv2
import numpy as np
paths_img_filtradas = ["imagenes_filtradas/early_blight/"
                       "imagenes_filtradas/late_blight/",
                       "imagenes_filtradas/leaf_miner/",
                       "imagenes_filtradas/leaf_mold/",
                       "imagenes_filtradas/mosaic_virus/",
                       "imagenes_filtradas/septoria/",
                       "imagenes_filtradas/spider_mites/",
                       "imagenes_filtradas/yellow_leaf_curl_virus/"
                       ]

def read_img(image_path):
    """Retorna una imagen en escala de grises"""
    # Cargar una imagen en color (RGB)
    img_rgb = cv2.imread(image_path)

    # Convertir la imagen a escala de grises
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    return img_gray

def ecualizadoRGB(image):

    # Realizar la ecualización del histograma en escala de grises
    equ_gray = cv2.equalizeHist(image)

    # Crear una imagen en RGB con el canal de intensidad ecualizado
    equ_rgb = cv2.cvtColor(equ_gray, cv2.COLOR_GRAY2BGR)

    return equ_gray


def log_transform(image, c=1):
    # Aplicar la transformación logarítmica
    #log_image = c * (np.log(image + 1))
    # Normalizar la imagen
    #log_image = cv2.normalize(log_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #return log_image
    # Clonar la imagen original
    result = image.copy()

    # Obtener dimensiones de la imagen
    rows, cols = image.shape

    # Aplicar la transformación logarítmica a cada píxel
    for i in range(rows):
        for j in range(cols):
                result[i, j] = 31 * np.log2(1 + image[i, j])

    return result

def gamma_transform(img, gamma=0.5):
    # Transformación gamma
    img_corrected = np.power(img / 255.0, gamma)
    img_corrected = np.uint8(img_corrected * 255)
    return img_corrected


def ecualizacion_histograma(img):
    equ = cv2.equalizeHist(img)
    res = np.hstack((img, equ))
    return equ


def gaussiano(img,sigma=0):
    # Aplicar el filtro gaussiano
    kernel_size = (5, 5)
    img_gaussian = cv2.GaussianBlur(img, kernel_size, sigma)
    return img_gaussian
