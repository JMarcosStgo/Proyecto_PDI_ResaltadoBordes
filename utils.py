import random

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

