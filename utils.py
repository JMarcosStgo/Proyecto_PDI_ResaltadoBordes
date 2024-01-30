import random
from preprocesamiento import *
from utils import *
from filtros_estadisticos import *
from filtros_suavizantes import *
from filtros_realzantes import *
from algoritmos_bordes import *
from algoritmos_umbralizacion import *
from descriptores_region import *
import matplotlib.pyplot as plt


paths_img_filtradas = ["imagenes_filtradas/early_blight/",
                       "imagenes_filtradas/late_blight/",
                       "imagenes_filtradas/leaf_miner/",
                       "imagenes_filtradas/leaf_mold/",
                       "imagenes_filtradas/mosaic_virus/",
                       "imagenes_filtradas/septoria/",
                       "imagenes_filtradas/spider_mites/",
                       "imagenes_filtradas/yellow_leaf_curl_virus/",
                       ]

paths = ["imagenes/early_blight/",
         "imagenes/late_blight/",
         "imagenes/leaf_miner/",
         "imagenes/leaf_mold/",
         "imagenes/mosaic_virus/",
         "imagenes/septoria/",
         "imagenes/spider_mites/",
         "imagenes/yellow_leaf_curl_virus/",
         ]


def generar_numeros(class_number=1):
    """
    Generar 5 numeros randoms para leer las imagenes y agregar su suta
    @ params: class_number 
    0:early_blight
    1:late_blight
    2:leaf_miner
    3:leaf_mold
    4:mosaic_virus
    5:septoria
    6:spider_mites
    7:yellow_leaf_curl_virus
    """
    imagenes = [paths[class_number] +
                str(random.randint(1, 100)) + ".jpg" for i in range(5)]
    return imagenes

def guardar_imagenes(class_number,path_img):
    """Guarda las imagenes en sus respectivas carpeta de las imagenes filtradas"""


# Función para visualizar imágenes en una fila horizontal
def visualizar_en_fila_horizontal(imagenes, titulo):
    num_imagenes = len(imagenes)
    
    fig, axs = plt.subplots(1, num_imagenes, figsize=(num_imagenes * 3, 3))
    fig.suptitle(titulo, fontsize=16)
    
    for i in range(num_imagenes):
        axs[i].imshow(imagenes[i], cmap='gray')
        axs[i].axis('off')
    
    plt.show()

# Función para visualizar imágenes en 4 filas
def visualizar_en_4_filas(imagenes, titulo):
    fig, axs = plt.subplots(4, 5, figsize=(15, 12))
    fig.suptitle(titulo, fontsize=16)
    
    for i in range(4):
        for j in range(5):
            axs[i, j].imshow(imagenes[i*5 + j], cmap='gray')
            axs[i, j].axis('off')
    
    plt.show()


# Función para aplicar filtros y visualizar en 4 filas
def aplicar_filtros_y_visualizar3(imagenes, equalizado_,otsu,otsu_multiple_,titulo_grafica,class_number=0):
    imagenes_origininales = [read_img(img) for img in imagenes]

    # Combina todas las imágenes en una sola lista
    todas_las_imagenes = imagenes_origininales + equalizado_ + otsu + otsu_multiple_

    # Visualizar imágenes en 4 filas
    visualizar_en_4_filas(todas_las_imagenes, "clase de la plaga: "+ titulo_grafica)
