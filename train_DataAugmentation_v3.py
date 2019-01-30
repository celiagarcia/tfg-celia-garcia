from __future__ import print_function

import os
import itertools
import random
from sklearn.model_selection import KFold
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator
from skimage import exposure
import matplotlib.pyplot as plt
from libtiff import TIFFimage
#import Image

from data import load_data

NB_EPOCH = 1 #3
BATCH_SIZE = 1 #4
image_rows = 256
image_cols = 256
list_predict = [] # Creo un array vacio para rellenarlo con las 18 predicciones.
smooth = 1.
NUM_VAL= 3

def equal_hist(img):

    # Estiramiento de contraste

    # min de imgs: -3.01762446888.
    # max de imgs: 29.2902694143.
    #print(np.dtype(np.amax(img)))
    p2, p98 = np.percentile(img, (random.randint(0, 10), random.randint(90,100)))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))

    return img_rescale

def data_generator_keras(x, y, batch_size, full_dataset, flip):


    # IMGS
    datagen_x = ImageDataGenerator(
        # Estandarizar los valores de pixeles en todo el conjunto de datos. Esto se denomina estandarizacion de caracteristicas.
        featurewise_center=True,    # Establece la media de entrada en 0 del conjunto de datos.
        featurewise_std_normalization=True, # Divide las entradas por la desviacion tipica del conjunto de datos.
        # --zca_epsilon,
        # zca_whitening = True,     # Menos redundancia en la imagen, resaltar mejor las estructuras y caracteristicas de la imagen para el algoritmo de aprendizaje.
        rotation_range=90,        # Rango de grados para rotaciones aleatorias
        width_shift_range=0.2,    # Desplazamientos aleatorios horizontales
        height_shift_range=0.2,   # Desplazamientos aleatorios horizontales
        shear_range=0.2,          # Angulo de corte en sentido antihorario en radianes
        zoom_range=0.2,           # Rango para zoom aleatorio. Float o [lower, upper] = [1-zoom_range, 1+zoom_range]
        horizontal_flip=flip,       # Voltea las entradas horizontalmente de forma aleatoria      
        vertical_flip=flip,         # Voltea las entradas verticalmente de forma aleatoria 
        fill_mode='nearest',        # Los puntos fuera de los limites de la entrada se rellenan segun el modo. "nearest":  aaaaaaaa|abcd|dddddddd
        preprocessing_function=equal_hist)

    # MASK
    datagen_y = ImageDataGenerator(
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=flip,
        vertical_flip=flip,
        fill_mode='nearest')

    if full_dataset is not None:
        datagen_x.fit(full_dataset)

    for batch in itertools.izip(
        datagen_x.flow(x, batch_size=batch_size, seed=1337),
        #datagen_x.flow(x, batch_size=batch_size, seed=1337, save_to_dir=os.path.join('DA'), save_prefix='imgsDA', save_format='png'),
        datagen_y.flow(y, batch_size=batch_size, seed=1337)
        ):
        yield batch

# Se le pasa el mask que ha creado el modelo, y el mask original, para ver como de bueno es el resultado
def dice_coef(y_true, y_pred):
    # El coeficiente dice compara dos planos, por eso cojo la segunda capa de los mask (capa celula)
    y_true = y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def model_evaluate(imgs, imgs_mask, train_index, test_index):
    model = get_unet()

    # Guarda el modelo despues de cada epoca
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    # Creo los subconjuntos de TRAIN, VALIDATION y TEST

    print('-'*30)
    print('Creating the subsets of train, validation and test...')
    print('-'*30)

    imgs_fulltrain = imgs[train_index] # (10, 256, 256, 1)
    imgs_mask_fulltrain = imgs_mask[train_index] # (10, 256, 256, 2)

    index = np.arange(len(imgs_fulltrain)) # [0,1,2...8,9]
    index_val = np.random.choice(index, NUM_VAL)
    index_train = np.delete(index, index_val)

    imgs_train = imgs_fulltrain[index_train]
    imgs_mask_train = imgs_mask_fulltrain[index_train]
    imgs_validation = imgs_fulltrain[index_val]
    imgs_mask_validation = imgs_mask_fulltrain[index_val]

    imgs_test = imgs[test_index]
    imgs_mask_test = imgs_mask[test_index] 

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit_generator(data_generator_keras(x=imgs_train, y=imgs_mask_train, batch_size=BATCH_SIZE, full_dataset=imgs, flip=True),
                epochs=NB_EPOCH,
                steps_per_epoch=4,
                verbose=2,
                callbacks=[model_checkpoint], 
                validation_data=(imgs_validation,imgs_mask_validation))

    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')
    print('-'*30)
    print('Evaluating masks on test data...')
    print('-'*30)

    # Devuelve el valor de perdida y los valores de metrica para el modelo en modo de prueba.
    imgs_mask_test_evaluate = model.evaluate(imgs_test, imgs_mask_test, verbose=1)




# -----------------------------------------
    # Guardo el resultado de la evaluacion de la imagen test
    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    # Con el modelo entrenado, devuelve la mask con la imagen que le metas.
    imgs_test = imgs[test_index]
    imgs_mask_test = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test.npy', imgs_mask_test) # (1, 256, 256, 2)


    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test, test_index):
        image = (image[:, :, 0] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_pred.png'), image)

        # -----matplotlib-------
        #print(image.shape)
        #plt.imshow(image[:,:,0],vmin=0,vmax=1)
        #plt.show()
# -----------------------------------------

    #data = np.zeros((1024,1024),np.uint16)
    #h,w = data.shape
    #for i in range(w):
    #    data[:,i] = np.arange(h)
    #tiff = TIFFimage(data, description='')
    #tiff.write_file('test_16bit.tif', compression='lzw')
    #flush the file to disk:
    #del tiff
#...............................................


    return imgs_mask_test_evaluate


def get_unet():

    print('-'*30)
    print('Creating and compiling model UNET...')
    print('-'*30)

    inputs = Input((image_rows, image_cols, 1)) # (256, 256, 1)
    # padding same: con relleno cero ???
    # Relu como funcion de activacion
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs) # (256, 256, 32)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1) # (256, 256, 32)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) # (128, 128, 32)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1) # (128, 128, 64)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2) # (128, 128, 64)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) # (64, 64, 64)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2) # (64, 64, 128)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3) # (64, 64, 128)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) # (32, 32, 128)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3) # (32, 32, 256)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4) # (32, 32, 256)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) # (16, 16, 256)

    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4) # (16, 16, 512)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5) # (16, 16, 512)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3) 
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # Usar Softmax para tener la distribucion de probabilidad en la direccion de profundidad (capas)
    # La salida es distribucion de probabilidad en dos capas "celula" y "fondo"
    conv10 = Conv2D(2, (1, 1), activation='softmax')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])


    # LOSS: La variable loss guarda el nodo que calcula el error de la red. Se compara la salida de la red, con la salida deseada con la cross entropy media.
    # categorical_crossentropy, por eso utilizo anteriormente to_categorical.

    # OPTIMIZER:  Optimizador elegido para minimizar el error iterativamente. En cada iteracion se ejecuta el optimizador, que a su vez recibe el error de la red,
    # que a su vez calcula las convoluciones y operaciones necesarias, que a su vez recibe una serie de imagenes de entrada (en ultima instancia el optimizador esta relacionado con toda la red, asi que solo hace falta ejecutar el nodo del optimizador para entrenar la red)
    # Mejorar los pesos.

    # METRICS: Lista de metricas que evaluara el modelo durante la capacitacion y las pruebas. 
    # Dice Coef evalua cuanto se parece la mask que se ha predicho con la mask real.
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=[dice_coef])

    return model

def train_and_predict():
    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)
    imgs, imgs_mask = load_data() # float64
    # min de imgs: -3.01762446888.
    # min de imgs mask: 0.0.
    # max de imgs: 29.2902694143.
    # max de imgs mask: 1.0.

    print('-'*30)
    print('Start Cross Validation...')
    print('-'*30)
    
    kf = KFold(n_splits=len(imgs))
    # Para cada una de las 11 iteraciones, dame los indices de train y test
    for train_index, test_index in kf.split(imgs): 

        # iteracion 1: train_index = [1 2 3 4 5 6 7 8 9 10] y test_index = [0]
        #La validacion cruzada es para valorar como de bien funciona el modelo, por eso en cada iteracion me creo un modelo de cero, no es para entrenar todos seguidos.
        #Dentro del bucle hay varios epochs. ??????
        #Se crean pesos nuevos desde 0.

        #Devuelve el valor de perdida y los valores de metrica para el modelo en modo de prueba.
        imgs_mask_test_evaluate = model_evaluate(imgs, imgs_mask, train_index, test_index) #[0.4318084716796875, 0.022709527984261513]
        list_predict.append(imgs_mask_test_evaluate[1])

        print('Model Evaluate: categorical_crossentropy = %f, dice_coef = %f' % (imgs_mask_test_evaluate[0],imgs_mask_test_evaluate[1]))
        print('Lista de las 11 imgs test: {0}.'.format(list_predict))


        break # Solo una iteracion para probar

    # El resultado final es la media de toda la lista (18 imagenes test)
    result = np.mean(list_predict)
    print('Resultado final: {0}.'.format(result))



if __name__ == '__main__':
    train_and_predict()
