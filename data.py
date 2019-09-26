from __future__ import print_function
#
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

path = 'raw/'

image_rows = 256
image_cols = 256

def create_data():
    data_path = os.path.join(path, '01/')
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
                img = imread(os.path.join(data_path, image_name_data), as_gray=True) #(832, 992), uint16, max:54528, min:3584
                
                #hist, hist_centers = exposure.histogram(img, nbins='auto')
                #fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
                #ax[0].imshow(img, interpolation='nearest', cmap=plt.cm.gray)
                #ax[0].axis('off')
                #ax[1].plot(hist_centers, hist, lw="2")
                #ax[1].set_title('Histogram of img read')
                #plt.tight_layout()
                #plt.show()
                #print('Max de la img recien leida: {0}.'.format(np.max(img)))
                #print('Min de la img recien leida: {0}.\n'.format(np.min(img)))


                #preserve_range: mantener el rango de valores original
                img = resize(img, (image_rows, image_cols), preserve_range=True) #(256, 256), float64, max:45412.0, min:4534.0
                img = np.reshape(img, (image_rows, image_cols, 1)) #(256, 256, 1), float64, max:45412.0, min:4534.0
            

                # ------------------- Analizo las mask ---------------------

                # * * * * * READ * * * * * * 

                img_mask = imread(os.path.join(mask_path, image_name_mask)) #(832, 992), uint16, max:11, min:0

                ##### histogramas read #####

                #print(np.histogram(img_mask))
                #a = np.hstack(img_mask)
                #plt.hist(a, bins='auto')  # arguments are passed to np.histogram
                #plt.title("Histogram read")
                #plt.show()
                # [772235,   4419,   9647,   2811,   6167,   3033,  11476,   4618, 3180,   7758],
                # [ 0. ,  1.1,  2.2,  3.3,  4.4,  5.5,  6.6,  7.7,  8.8,  9.9, 11. ]
                #hist, hist_centers = exposure.histogram(img_mask, nbins='auto')
                #fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
                #ax[0].imshow(img_mask, interpolation='nearest', cmap=plt.cm.gray)
                #ax[0].axis('off')
                #ax[1].plot(hist_centers, hist, lw="2")
                #ax[1].set_title('Histogram of mask read')
                #plt.tight_layout()
                #plt.show()
                #[770804,   1431,   4419,   9647,   2811,   6167,   3033,  11476, 4618,   3180,   4255,   3503],
                # [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]


                # * * * * *  RESIZE * * * * * *

                img_mask = resize(img_mask, (image_rows, image_cols), preserve_range=True) #(256, 256), float64
                #print('Max de la mask depsues del resize 256x256: {0}.'.format(np.max(img_mask))) 
                #print('Min de la mask depsues del resize 256x256: {0}.\n'.format(np.min(img_mask))) 

                ##### histogramas resize #####

                #print(np.histogram(img_mask))
                #a = np.hstack(img_mask)
                #plt.hist(a, bins='auto')  # arguments are passed to np.histogram
                #plt.title("Histogram resize")
                #plt.show()
                #[61200,   478,   760,   248,   490,   258,   918,   348,   239, 597],
                #[ 0. ,  1.1,  2.2,  3.3,  4.4,  5.5,  6.6,  7.7,  8.8,  9.9, 11. ]
                #hist, hist_centers = exposure.histogram(img_mask, nbins='auto')
                #fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
                #ax[0].imshow(img_mask, interpolation='nearest', cmap=plt.cm.gray)
                #ax[0].axis('off')
                #ax[1].plot(hist_centers, hist, lw="2")
                #ax[1].set_title('Histogram of mask resize')
                #plt.tight_layout()
                #plt.show()
                # [61007,   226,   109,   367,   723,    33,   225,   482,    12, 244,   909,    15,   343,   241,     3,   329,   268],
                # [ 0.32352941,  0.97058824,  1.61764706,  2.26470588,  2.91176471, 3.55882353,  4.20588235,  4.85294118,  5.5 ,  6.14705882, 6.79411765,  7.44117647,  8.08823529,  8.73529412,  9.38235294, 10.02941176, 10.67647059]))


                # * * * * *  BINARIZED * * * * * *

                # Las mascaras de segmentacion tienen un numero diferente para cada celula
                # Si se especifica un intervalo de [0, 1], los valores menores que 0 se convierten en 0 y los valores mayores que 1 se convierten en 1.
                #img_mask = np.clip(img_mask, 0, 1) #(256, 256), float64
                img_mask = 1.0 * (img_mask > 0.2)
                #print('Max de la mask depsues del binarized [0,1]: {0}.'.format(np.max(img_mask)))
                #print('Min de la mask depsues del binarized [0,1]: {0}.\n'.format(np.min(img_mask)))

                ##### histogramas binarized #####

                #a = np.hstack(img_mask)
                #plt.hist(a, bins='auto')  # arguments are passed to np.histogram
                #plt.title("Histogram binarized, threshold 0.2")
                #plt.show()
                #print(np.histogram(img_mask))
                # [60889,     0,     0,     0,     0,     0,     0,     0,     0, 4647],
                # [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ]

                #print(exposure.histogram(img_mask, nbins='auto'))
                #hist, hist_centers = exposure.histogram(img_mask, nbins='auto')
                #fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
                #ax[0].imshow(img_mask, interpolation='nearest', cmap=plt.cm.gray)
                #ax[0].axis('off')
                #ax[1].plot(hist_centers, hist, lw="2")
                #ax[1].set_title('Histogram of mask binarized, threshold 0.2')
                #plt.tight_layout()
                #plt.show()
                # [60842,    20,    13,    25,    23,    17,    20,     7,    10, 21,     9,     9,    19,    16,    10,    19,  4456],
                # [0.02941176, 0.08823529, 0.14705882, 0.20588235, 0.26470588,0.32352941, 0.38235294, 0.44117647, 0.5 , 0.55882353,0.61764706, 0.67647059, 0.73529412, 0.79411765, 0.85294118,0.91176471, 0.97058824]))



                # * * * * *  CATEGORICAL * * * * * *

                # Construccion de etiquetas de entrenamiento. Convierto las mascaras de un canal a dos canales, uno por cada etiqueta (fondo y celula).
                categorical_labels = to_categorical(img_mask, num_classes = None) #(256, 256, 2), float32


                ##### histogramas categorical #####


                #print(np.histogram(categorical_labels[:,:,0]))
                #[ 4647,     0,     0,     0,     0,     0,     0,     0,     0, 60889],
                # [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], type=float32))
                #a = np.hstack(categorical_labels[:,:,0])
                #plt.hist(a, bins='auto')  # arguments are passed to np.histogram
                #plt.title("Histogram categorical_labels, label 1")
                #plt.show()


                #print(np.histogram(categorical_labels[:,:,1]))
                #[60889,     0,     0,     0,     0,     0,     0,     0,     0, 4647]),
                # [0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ], dtype=float32))
                #a = np.hstack(categorical_labels[:,:,1])
                #plt.hist(a, bins='auto')  # arguments are passed to np.histogram
                #plt.title("Histogram categorical_labels, label 0")
                #plt.show()



   
                #print(exposure.histogram(categorical_labels[:,:,1], nbins='auto'))
                #hist, hist_centers = exposure.histogram(categorical_labels[:,:,1], nbins='auto')
                #fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
                #ax[0].imshow(categorical_labels[:,:,1], interpolation='nearest', cmap=plt.cm.gray)
                #ax[0].axis('off')
                #ax[1].plot(hist_centers, hist, lw="2")
                #ax[1].set_title('Histogram categorical_labels, label 0')
                #plt.tight_layout()
                #plt.show()

                #[60889,     0,     0,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0,     0,  4647]),
                # [0.02941176, 0.0882353 , 0.14705883, 0.20588236, 0.2647059 , 0.32352942, 0.38235295, 0.44117647, 0.5       , 0.5588236 , 0.61764705, 0.67647064, 0.7352941 , 0.7941177 , 0.85294116, 0.91176474, 0.9705882 ], dtype=float32))



                # No hace falta hacer reshape, categorical ya es (256,256,2)
                # * * * * *  RESHAPE * * * * * *
                # Pero categorial_labels tiene que ser de la forma 256x256x2, hago un reshape
                #categorical_labels = np.reshape(categorical_labels, (image_rows, image_cols, 2))

                print('Tipo de img: {0}.'.format(img[:, :, 0].dtype))
                print('Tipo de img_mask: {0}.'.format(categorical_labels[:, :, 0].dtype))


                imgs[i] = img
                imgs_mask[i] = categorical_labels
                break # Para que siga con la siquiente mask
                
        i += 1


    hist, hist_centers = exposure.histogram(imgs[3][:,:,0], nbins='auto')
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(imgs[3][:,:,0], interpolation='nearest', cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[1].plot(hist_centers, hist, lw="2")
    ax[1].set_title('Histogram before normalization, image 3')
    plt.tight_layout()
    plt.show()
    
    # Normalization
    mean = np.mean(imgs) # Devuelve el promedio de los elementos de la matriz.
    std = np.std(imgs) # Devuelve la desviacion tipica, una medida de la extension de una distribucion, de los elementos de la matriz.
    imgs -= mean
    imgs /= std
    print('mean imgs: {0}.'.format(np.mean(imgs)))
    print('desv tipica imgs: {0}.'.format(np.std(imgs)))

    hist, hist_centers = exposure.histogram(imgs[3][:,:,0], nbins='auto')
    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    ax[0].imshow(imgs[3][:,:,0], interpolation='nearest', cmap=plt.cm.gray)
    ax[0].axis('off')
    ax[1].plot(hist_centers, hist, lw="2")
    ax[1].set_title('Histogram after normalization, image 3')
    plt.tight_layout()
    plt.show()
    
    print('Loading done.')
    np.save('imgs.npy', imgs) # type imgs: float64
    np.save('imgs_mask.npy', imgs_mask) # type imgs_mask float64
    print('Saving to .npy files done.')

def load_data():
    imgs = np.load('imgs.npy')
    imgs_mask = np.load('imgs_mask.npy')
    return imgs, imgs_mask


if __name__ == '__main__':
    create_data()



