from preprocesamiento import *
from utils import *
from filtros_estadisticos import *
from filtros_suavizantes import *
from filtros_realzantes import *
from algoritmos_bordes import *
import matplotlib.pyplot as plt

def main():
    # Filtros necesarios para resaltar los bordes de las hojas de la clase 0
    #equalizado_ = [ canny(gaussiano(sobel(laplaciano(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=.4))))))) for x in imagenes]
    
    # Filtros necesarios para resaltar los bordes de las hojas de la clase 1
    #equalizado_ = [ canny(convolucion_copia_borde(ecualizadoRGB(gaussiano(mediana(read_img(x),ksize=9))))) for x in imagenes]
    

    imagenes= generar_numeros(class_number=2)
    # Filtros necesarios para resaltar los bordes de las hojas de la clase 2
    equalizado_ = [canny(gaussiano(gamma_transform(read_img(x)),sigma=0.3)) for x in imagenes] 
    

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 3
    #equalizado_ = [ canny(highboost(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=0.5),sigma=6)),k=4)) for x in imagenes]

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 4
    #equalizado_ = [ canny(convolucion_copia_borde(mediana(gaussiano(read_img(x))))) for x in imagenes]

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 5
    #equalizado_ = [ canny(gaussiano(convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(read_img(x))))))) for x in imagenes]

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 6
    #equalizado_ = [ canny((convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(gamma_transform(read_img(x))),sigma=1),ksize=7)))) for x in imagenes]

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 7
    #equalizado_ = [ canny((convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(gamma_transform(read_img(x),gamma=0.33)),sigma=1),ksize=7)))) for x in imagenes]
    #equalizado_ = [ canny((convolucion_copia_borde(ecualizadoRGB((mediana(gamma_transform(read_img(x),gamma=2))))))) for x in imagenes]
    #equalizado_ = [ (mediana(ecualizacion_histograma(read_img(x)))) for x in imagenes]


    #lectura de las imagenes originales y se agrega a la lista de imagenes para mostrarlas
    imagenes = [cv2.imread(img_) for img_ in imagenes]
    equalizado_ =equalizado_ + imagenes
    


    # Crear una figura con 2 filas y 5 columnas
    fig, axs = plt.subplots(2, 5)
    # Mostrar las imágenes en la figura
    for i in range(2):
        for j in range(5):
            if i ==0:
                axs[i, j].imshow(equalizado_[i * 5 + j], cmap='gray')
            else:
                axs[i, j].imshow(equalizado_[i * 5 + j])
                
            axs[i, j].axis('off')

    plt.show()


main()


