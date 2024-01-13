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