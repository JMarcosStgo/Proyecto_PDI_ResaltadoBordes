import numpy as np
import cv2
import matplotlib.pyplot as plt


def convolucion(img, filtro_mask=1):
    # Definir una matriz de convolución (por ejemplo, un filtro de detección de bordes)
    kernel = np.array([[-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0],
                       [-1.0, 0.0, 1.0]])
    # filtro de caja
    kernel2 = np.array([[1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0]])
    # promedio pesado
    kernel3 = np.array([[1.0, 2.0, 1.0],
                        [2.0, 4.0, 2.0],
                        [1.0, 2.0, 1.0]])

    if filtro_mask == 1:        
        # Realizar la convolución
        resultado_conv = cv2.filter2D(img, -1, kernel)
    elif filtro_mask == 2:
        # Realizar la convolución
        resultado_conv = cv2.filter2D(img, -1, kernel2)
    else:
        # Realizar la convolución
        resultado_conv = cv2.filter2D(img, -1, kernel3)

    # Normalizar el resultado para que esté en el rango 0-255
    resultado_normalizado = cv2.normalize(
        resultado_conv, None, 0, 255, cv2.NORM_MINMAX)
    return resultado_normalizado


def convolucion2(img):
    # Convertir a tipo de dato double
    J = img.astype(float)

    # Filtro de caja de 3x3
    w = np.ones((3, 3), dtype=float)

    # Convolución
    Convl = cv2.filter2D(J, -1, w, borderType=cv2.BORDER_CONSTANT) * (1/9)
    return Convl


def convolucion_copia_borde(img):
    # Convertir a tipo de dato double
    J = img.astype(float)

    # Filtro de caja de 3x3
    w = np.ones((3, 3), dtype=float)

    # Convolución, copia borde de la imagen
    ConvlR = cv2.filter2D(J, -1, w, borderType=cv2.BORDER_REPLICATE) * (1/9)
    return ConvlR


def correlacion(img):
    # Convertir a tipo de dato double
    J = img.astype(float)

    # Filtro de caja de 3x3
    w = np.ones((3, 3), dtype=float)

    # Correlación
    Corrl = cv2.filter2D(J, -1, w) * (1/9)
    return Corrl


# Cargar una imagen en color (RGB)
#img_rgb = cv2.imread("imagenes/leaf_miner/85.jpg")
# Convertir la imagen a escala de grises
#img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
#plt.imshow(img_gray,cmap='gray')
#plt.axis('off')  # Desactiva los ejes
#plt.show()
#gaus = convolucion(img_gray,filtro_mask=3)
#plt.imshow(gaus,cmap='gray')
#plt.axis('off')  # Desactiva los ejes
#plt.show()