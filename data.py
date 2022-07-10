from __future__ import print_function
#
# Celia García Fernández
# data.py para leer y cargar los datos (images y mask) en formato .npy
# indicar en "path" el directorio de dichos datos
#
# https://academic.oup.com/bioinformatics/article/30/11/1609/283435
# https://www.nature.com/articles/nmeth.4473
#
#
import os
import numpy as np
from skimage.io import imsave, imread, imshow
from skimage.transform import resize
from skimage import data, exposure, img_as_float
from keras.utils.np_utils import to_categorical 
import matplotlib.pyplot as plt

path = 'rawmix/'

image_rows = 256
image_cols = 256

def create_data():
    data_path = os.path.join(path, '01')
    mask_path = os.path.join(path, '01_GT/SEG')
    images_mask_list = os.listdir(mask_path) #lista de nombres de mask
    images_data_list = os.listdir(data_path) #lista de nombres de data

    total = len(images_mask_list)

    imgs = np.ndarray((total, image_rows, image_cols, 1)) #(11, 256, 256, 1)  # por defecto son float
    imgs_mask = np.ndarray((total, image_rows, image_cols, 2)) #(11, 256, 256, 2)

    i = 0
    print('-'*30)
    print('Creating images...')
    print('-'*30)
    for image_name_mask in images_mask_list:
        id_mask = image_name_mask[-6:]
        print('IMAGEN: {0}.\n'.format(id_mask))
        for image_name_data in images_data_list:
            if id_mask in image_name_data: # Compruebo que id_mask coincide con id_imagen.
                # as_gray: True convierte las imagenes en color a escala de grises (flotantes de 64 bits). Las imagenes que ya estan en formato de escala de grises no se convierten.
                img = imread(os.path.join(data_path, image_name_data), as_gray=True) #(832, 992) o (782, 1200), uint16
                
                #preserve_range: mantener el rango de valores original
                img = resize(img, (image_rows, image_cols), preserve_range=True) #(256, 256), float64, max:45412.0, min:4534.0
                img = np.reshape(img, (image_rows, image_cols, 1)) #(256, 256, 1), float64, max:45412.0, min:4534.0


                # ------------------- Analizo las mask ---------------------

                # READ
                #imsave(os.path.join('data', str(id_mask) + '_' + '.png'), img_mask)
                img_mask = imread(os.path.join(mask_path, image_name_mask)) #(832, 992), uint16, max:11, min:0

                # BINARIZED
                # Las mascaras de segmentacion tienen un numero diferente para cada celula
                # Si se especifica un intervalo de [0, 1], los valores menores que 0 se convierten en 0 y los valores mayores que 1 se convierten en 1.
                #img_mask = np.clip(img_mask, 0, 1) #(256, 256), float64
                img_mask = 1.0 * (img_mask > 0.2)

                # RESIZE
                img_mask = resize(img_mask, (image_rows, image_cols), preserve_range=True) #(256, 256), float64
                img_mask = 1.0 * (img_mask > 0.2)
                #imsave(os.path.join('data', str(id_mask) + '_resize_despuesdebinarize_resize_y binarize' + '.png'), img_mask)

                # CATEGORICAL
                # Construccion de etiquetas de entrenamiento. Convierto las mascaras de un canal a dos canales, uno por cada etiqueta (fondo y celula).
                categorical_labels = to_categorical(img_mask, num_classes = None) #(256, 256, 2), float32


                imgs[i] = img
                imgs_mask[i] = categorical_labels
                break # Para que siga con la siquiente mask
                
        i += 1


    # Normalization
    mean = np.mean(imgs) # Devuelve el promedio de los elementos de la matriz.
    std = np.std(imgs) # Devuelve la desviacion tipica, una medida de la extension de una distribucion, de los elementos de la matriz.
    imgs -= mean
    imgs /= std
    #print('mean imgs: {0}.'.format(np.mean(imgs)))
    #print('desv tipica imgs: {0}.'.format(np.std(imgs)))


    print('Loading done.')
    np.save('imgs_mix.npy', imgs) # type imgs: float64
    np.save('imgs_mask_mix.npy', imgs_mask) # type imgs_mask float64
    print('Saving to .npy files done.')

def load_data_2():
    imgs = np.load('imgs2.npy')
    imgs_mask = np.load('imgs_mask2.npy')
    return imgs, imgs_mask

def load_data_1():
    imgs = np.load('imgs.npy')
    imgs_mask = np.load('imgs_mask.npy')
    return imgs, imgs_mask

def load_data_mix():
    imgs = np.load('imgs_mix.npy')
    imgs_mask = np.load('imgs_mask_mix.npy')
    return imgs, imgs_mask


if __name__ == '__main__':
    create_data()



