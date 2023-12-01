import cv2
def mediana(img,ksize=5):
    # Aplica el filtro de mediana
    img_median = cv2.medianBlur(img, ksize)
    return img_median