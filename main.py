from preprocesamiento import *
from utils import *
from filtros_estadisticos import *
from filtros_suavizantes import *
from filtros_realzantes import *
from algoritmos_bordes import *
from algoritmos_umbralizacion import *
from descriptores_region import *

import matplotlib.pyplot as plt

def main():
    # Cambiar el parametro de la funcion para seleecionar otra plaga
    imagenes= generar_numeros(class_number=0)
    
     # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 0
    equalizado_ = [ canny(gaussiano(sobel(laplaciano(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=.4))))))) for x in imagenes]
    # Filtro ostu simple
    otsu = [otsu_thresholding(gaussiano(sobel(laplaciano(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=.4)))))))for x in imagenes]
    # Filtro ostu multiple
    otsu_multiple_ = [otsu_multiple(gaussiano(sobel(laplaciano(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=.4)))))),num_classes=4)for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Early blight")
   
    # Lectura de las imagenes
    imagenes= generar_numeros(class_number=1)
    
    # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 1
    equalizado_ = [ canny(convolucion_copia_borde(ecualizadoRGB(gaussiano(mediana(read_img(x),ksize=9))))) for x in imagenes]
    # Filtro ostu simple
    otsu = [otsu_thresholding(convolucion_copia_borde(ecualizadoRGB(gaussiano (mediana(read_img(x)))))) for x in imagenes]
    # Filtro ostu multiple
    otsu_multiple_ = [otsu_multiple(convolucion_copia_borde(mediana(gaussiano ((read_img(x))))),num_classes=2) for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Late blight")
    


    imagenes= generar_numeros(class_number=2)

    # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 2
    equalizado_ = [canny(gaussiano(gamma_transform(read_img(x)),sigma=0.3)) for x in imagenes] 
    # Filtro ostu simple
    otsu = [otsu_thresholding(gaussiano(gamma_transform(read_img(x)),sigma=0.3))for x in imagenes]
    # Filtros otsu multiple
    otsu_multiple_ = [otsu_multiple(gaussiano(gamma_transform(read_img(x)),sigma=0.3),num_classes=3)for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Leaf miner")


    imagenes= generar_numeros(class_number=3)

    # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 3
    equalizado_ = [ canny(highboost(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=0.5),sigma=6)),k=4)) for x in imagenes]
    # Filtro ostu simple
    otsu = [otsu_thresholding(highboost(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=0.5),sigma=6)),k=4))for x in imagenes]
    # Filtro ostu multiple
    otsu_multiple_ = [otsu_multiple((highboost(convolucion_copia_borde(gaussiano(gamma_transform(read_img(x),gamma=0.5),sigma=6)),k=4)),num_classes=3)for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Leaf mold")
    

    imagenes= generar_numeros(class_number=4)

    # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)
    
    # Filtros necesarios para resaltar los bordes de las hojas de la clase 4
    equalizado_ = [ canny(convolucion_copia_borde(mediana(gaussiano(read_img(x))))) for x in imagenes]
    # Filtro ostu simple
    otsu = [otsu_thresholding(convolucion_copia_borde(mediana(gaussiano(read_img(x))))) for x in imagenes]
    # Filtro ostu multiple
    otsu_multiple_ = [otsu_multiple(convolucion_copia_borde(mediana(gaussiano(read_img(x)))),num_classes=3)for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Mosaic virus")
    

    
    imagenes= generar_numeros(class_number=5)

    # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 5
    equalizado_ = [ canny(gaussiano(convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(read_img(x))))))) for x in imagenes]
    # Filtro ostu simple
    otsu = [otsu_thresholding(gaussiano(convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(read_img(x)))))))for x in imagenes]
    # Filtro ostu multiple
    otsu_multiple_ = [otsu_multiple(gaussiano(convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(read_img(x)))))),num_classes=2)for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Septoria")
    


    imagenes= generar_numeros(class_number=6)

    # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 6
    equalizado_ = [ canny((convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(gamma_transform(read_img(x))),sigma=1),ksize=7)))) for x in imagenes]
    # Filtro ostu simple
    otsu = [otsu_thresholding(((mediana(gaussiano(((read_img(x))))))))for x in imagenes]
    # Filtro ostu multiple
    otsu_multiple_ = [otsu_multiple((mediana(gaussiano(read_img(x)))),num_classes=4) for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Spider mites")


    imagenes= generar_numeros(class_number=7)

    # Muestra los descriptores 
    compactness, hu_moments = calculate_compactness_and_hu_moments(imagenes[4])
    print_results_compactness_hu(compactness, hu_moments)
    indices_and_descriptors = calculate_indices_and_texture_descriptors(imagenes[4])
    print_results_texture_descriptors(indices_and_descriptors)

    # Filtros necesarios para resaltar los bordes de las hojas de la clase 7
    #equalizado_ = [ canny((convolucion_copia_borde(mediana(gaussiano(ecualizadoRGB(gamma_transform(read_img(x),gamma=0.33)),sigma=1),ksize=7)))) for x in imagenes]
    #equalizado_ = [ canny((convolucion_copia_borde(ecualizadoRGB((mediana(gamma_transform(read_img(x),gamma=2))))))) for x in imagenes]
    equalizado_ = [ (mediana(ecualizacion_histograma(read_img(x)))) for x in imagenes]
    # Filtro ostu simple
    otsu = [otsu_thresholding(mediana(ecualizacion_histograma(read_img(x))))for x in imagenes]
    # Filtro ostu multiple
    otsu_multiple_ = [otsu_multiple(mediana(ecualizacion_histograma(read_img(x))),num_classes=2)for x in imagenes]
    # Muestra las graficas
    aplicar_filtros_y_visualizar3(class_number=0,equalizado_=equalizado_,otsu=otsu,otsu_multiple_=otsu_multiple_,titulo_grafica="Yellow leaf curl virus")




main()