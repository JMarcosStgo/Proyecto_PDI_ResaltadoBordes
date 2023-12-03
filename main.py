from preprocesamiento import *
from utils import *
from filtros_estadisticos import *
from filtros_suavizantes import *
from filtros_realzantes import *
from algoritmos_bordes import *
import matplotlib.pyplot as plt

CLASS_NAMES = [
        'EARLY_BLIGHT',
        'LATE_BLIGHT',
        'LEAF_MINER',
        'LEAF_MOLD',
        'MOSAIC_VIRUS',
        'SEPTORIA',
        'SPIDER_MITES',
        'YELLOW_LEAF_CURL_VIRUS'
    ]


def apply_filters(class_number, filter_function):
    # Obtener imágenes
    imagenes = generar_numeros(class_number=class_number)
    # Aplicar filtros específicos para la clase
    equalizado_ = [filter_function(read_img(x)) for x in imagenes]
    # Agregar imágenes originales a la lista para mostrarlas
    imagenes = [cv2.imread(img_) for img_ in imagenes]
    equalizado_ = equalizado_ + imagenes
    # Crear una figura con 2 filas y 5 columnas
    fig, axs = plt.subplots(2, 5)
    # Agregar un título general a la figura
    fig.suptitle(CLASS_NAMES[class_number])
    # Mostrar las imágenes en la figura
    for i in range(2):
        for j in range(5):
            if i == 0:
                axs[i, j].imshow(equalizado_[i * 5 + j], cmap='gray')
            else:
                axs[i, j].imshow(equalizado_[i * 5 + j])
            axs[i, j].axis('off')
    # Mostrar la figura
    plt.show()

def main():
    # Definir nombres de clases
    # Aplicar filtros para cada clase
    for class_number, class_name in enumerate(CLASS_NAMES):
        if class_number == 0:
            filter_function = lambda x: canny(gaussiano(sobel(laplaciano(convolucion_copia_borde(gaussiano(gamma_transform(x, gamma=0.4)))))))
        elif class_number == 1:
            filter_function = lambda x: canny(convolucion_copia_borde(ecualizadoRGB(gaussiano(mediana(x, ksize=9)))))
        elif class_number == 2:
            filter_function = lambda x: canny(gaussiano(gamma_transform(x), sigma=0.3))
        elif class_number == 3:
            filter_function = lambda x: canny(highboost(convolucion_copia_borde(gaussiano(gamma_transform(x, gamma=0.5), sigma=6)), k=4))
        elif class_number == 4:
            filter_function = lambda x: canny(convolucion_copia_borde(mediana(gaussiano(x))))
        elif class_number == 5:
            filter_function = lambda x: canny(gaussiano(convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(x))))))
        elif class_number == 6:
            filter_function = lambda x: canny((convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(gamma_transform(x))), sigma=1), ksize=7)))
        elif class_number == 7:
            filter_function = lambda x: canny((convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(gamma_transform(x), gamma=0.33)), sigma=1), ksize=7)))
            #canny((convolucion_copia_borde(ecualizadoRGB((mediana(gamma_transform(read_img(x),gamma=2)))))))
            #(mediana(ecualizacion_histograma(read_img(x))))

        # Aplicar la función con los filtros para la clase actual
        apply_filters(class_number, filter_function)

if __name__ == "__main__":
    main()


