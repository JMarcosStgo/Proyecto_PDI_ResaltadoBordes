import cv2
import numpy as np
def canny(img):
    # Aplicar el detector de bordes Canny
    #edges = cv2.Canny(img, threshold1=50, threshold2=100)

    # Buscar los contornos
    #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img = np.uint8(img)
    # Aplica el algoritmo de Canny
    print(img.shape,type(img))
    edges = cv2.Canny(img,100,200)
    
    # Dibujar los contornos en la imagen original
    #cv2.drawContours(img, contours, -1, (0, 255, 0), 2)

    return edges

def sobel(image):
    # Aplicar el operador Sobel en las direcciones x e y
    sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

    # Tomar el valor absoluto del gradiente en ambas direcciones
    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    # Combinar los resultados en una imagen final
    sobel_resultado = cv2.bitwise_or(sobelX, sobelY)
    return sobel_resultado