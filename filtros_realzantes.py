import numpy as np
import cv2
import matplotlib.pyplot as plt


def laplaciano(img,mask=1,c=1):
    # Convertir a tipo de dato double
    J = img.astype(float)

    # Filtros Laplacianos
    L1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    L2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    L3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    
    # Aplicar correlación con los filtros Laplacianos
    if mask==1: 
        corrl = cv2.filter2D(J, -1, L1)
    elif mask ==2:
        corrl = cv2.filter2D(J, -1, L2)
    else:
        corrl = cv2.filter2D(J, -1, L3)

    # Aplicar los filtros
    # Mascaras del Laplaciano con centro positivo si c =1, si c= -1 es con centro negativo 
    FL1 = J + c * corrl
    return FL1

def highboost(img,k=1):
    # Convertir a tipo de dato double
    J = img.astype(float)

    # Filtro de caja de 3x3
    w = np.ones((3, 3), dtype=float)

    # Convolución, copia borde de la imagen
    P1ConvlR = cv2.filter2D(J, -1, w, borderType=cv2.BORDER_REPLICATE) * (1/9)

    # Calcular la máscara
    P2Mask = J - P1ConvlR

    # Highboost
    Highboost = J + k * P2Mask
    return Highboost
    
def gradiente_laplaciano(img,c=1,mask=3):
    # Convertir a tipo de dato double
    J = img.astype(float)

    # Realzar imagen con el filtro Laplaciano
    #L3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    # Filtros Laplacianos
    L1 = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    L2 = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    L3 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    
    
    # Aplicar correlación con los filtros Laplacianos
    if mask==1: 
        L3 = L1
    elif mask ==2:
        L3 = L2
    else:
        L3 = L3


    
    Corrl3 = cv2.filter2D(J, -1, L3)
    P1FL3 = J + c * Corrl3

    # Magnitud del gradiente
    x = np.array([[-1, 2, -1], [0, 0, 0], [1, 2, 1]])
    y = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Px = cv2.filter2D(J, -1, x, borderType=cv2.BORDER_REPLICATE)
    Py = cv2.filter2D(J, -1, y, borderType=cv2.BORDER_REPLICATE)
    Mag = np.sqrt(Px**2 + Py**2)

    # Suavizado de magnitud
    w = np.ones((3, 3), dtype=float)
    Suav = cv2.filter2D(Mag, -1, w, borderType=cv2.BORDER_REPLICATE) * (1/9)
     
    # Mascara = Imagen realzada * Magnitud suavizada
    Mask = P1FL3 * Suav

    # Máscara + imagen original
    GradLap = J + Mask
    return GradLap

def gaussiano_pasa_altas(img):
    # Aplicar filtro gaussiano de pasa-altas
    kernel_size = 5
    sigma = 1.0
    img_gaussian = cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma) - img
    return img_gaussian

