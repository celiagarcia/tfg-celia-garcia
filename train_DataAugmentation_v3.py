from __future__ import print_function

#
#
# https://academic.oup.com/bioinformatics/article/30/11/1609/283435
# https://www.nature.com/articles/nmeth.4473
#
#


import os
import itertools

import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from libtiff import TIFFimage
from time import time
from datetime import datetime
from skimage.transform import resize
from skimage.io import imsave
from skimage import exposure
import numpy as np
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import backend as K
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator
#import scripts
from data import load_data
from model_unet import get_unet
import tensorflow as tf


NB_EPOCH = 100 # por ejemplo
#train 7 y val 3. batch_size = train_size = 7
BATCH_SIZE = 10
image_rows = 256
image_cols = 256
list_predict = []
NUM_VAL= 3 # numero de imagenes de validacion
CONT_CROSSVALIDATION = 0



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
        # Ajusta el generador de datos a algunos datos de muestra.
        # Calcula las estadisticas de datos internos relacionadas con las transformaciones dependientes de los datos, basadas en una matriz de datos de muestra.
        # Solo se requiere si featurewise_centero featurewise_std_normalizationo zca_whiteningestan establecidos en True.
        datagen_x.fit(full_dataset)

    # izip('ABCD', 'xy') --> Ax By
    for batch in itertools.izip(
        # flow: Toma datos y matrices de etiquetas, genera lotes de datos aumentados.
        # Genera imagenes
        datagen_x.flow(x, batch_size=batch_size, seed=1337),
        #datagen_x.flow(x, batch_size=batch_size, seed=1337, save_to_dir=os.path.join('DA'), save_prefix='imgsDA', save_format='png'),
        # Genera etiquetas
        datagen_y.flow(y, batch_size=batch_size, seed=1337)
        ):
        # flow devuelve un resultado de Iterator tuplas de (x, y) donde x es una matriz numpy de datos de imagen
        # (en el caso de una sola entrada de imagen) o una lista de matrices numpy (en el caso de entradas adicionales)
        # y y es una matriz numpy de etiquetas correspondientes. 
        yield batch


def model_evaluate(imgs, imgs_mask, train_index, test_index):
    
    print('-'*30)
    print('Creating the subsets of train, validation and test...')
    print('-'*30)
    imgs_fulltrain = imgs[train_index] # (10, 256, 256, 1)
    imgs_mask_fulltrain = imgs_mask[train_index] # (10, 256, 256, 2)

    index_train = np.arange(len(imgs_fulltrain)) # [0,1,2...8,9]
    print('------ INDEX: ', index_train)

    # sacar 3 imagenes de validacion dentro de las de fulltrain. Como no quiero que se repitan, segun voy escogiendo una, la voy eliminando.
    index_val = np.random.choice(index_train, 1)
    print('------ VAL_1: ', index_val)

    index_train = np.setdiff1d(index_train, index_val)
    print('------ INDEX - VAL_1: ', index_train)
    
    index_val = np.append(index_val, np.random.choice(index_train, 1))
    print('------ VAL: ', index_val)

    index_train = np.setdiff1d(index_train, index_val)
    print('------ INDEX - VAL_1 - VAL 2: ', index_train)
    
    index_val = np.append(index_val, np.random.choice(index_train, 1))
    print('------ VAL: ', index_val)
    
    index_train = np.setdiff1d(index_train, index_val)
    print('------ INDEX - VAL_1 - VAL 2 - VAL 3: ', index_train)
    
    #index_train = np.delete(index, index_val)
    
    
    print('------ ESTO ES INDEX VAL: ', index_val,' ----------')
    print('------ ESTO ES INDEX TRAIN: ', index_train,' ----------')


    imgs_train = imgs_fulltrain[index_train]
    imgs_mask_train = imgs_mask_fulltrain[index_train]
    imgs_validation = imgs_fulltrain[index_val]
    imgs_mask_validation = imgs_mask_fulltrain[index_val]

    imgs_test = imgs[test_index]
    imgs_mask_test = imgs_mask[test_index]
    
    # TensorBoard, representacion de imagenes y mascaras de entrenamiento
    #images_tb = np.reshape(imgs[0:11], (-1, 256, 256, 1))
    #masks_tb = np.reshape(imgs_mask[0:11], (-1, 256, 256, 2))
    #logdir = "logs/images/{}".format("images.{}".format(int(time())))
    # Creates a file writer for the log directory.
    #file_writer = tf.summary.create_file_writer(logdir)
    # Using the file writer, log the reshaped image.
    #with file_writer.as_default():
        #tf.summary.image("11 images - cross validation iteration {}".format(CONT_CROSSVALIDATION), images_tb, max_outputs=11, step=0)  
        #tf.summary.image("11 mask - cross validation iteration {}".format(CONT_CROSSVALIDATION), masks_tb, max_outputs=11, step=0)   
    

    model = get_unet()
    #tensorboard = TensorBoard(log_dir="logs/fit/{}".format("fit.{}".format(datetime.now().strftime("%Y%m%d-%H%M%S"))), histogram_freq=1, batch_size=BATCH_SIZE, update_freq='epoch')
    #tensorboard = TensorBoard(log_dir="logs/fit/{}".format("fit.{}".format(int(time()))), histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    
    # perfil de lote: Perfile el lote para muestrear las caracteristicas de calculo. Por defecto, perfilara el segundo lote. Establezca profile_batch = 0 para deshabilitar la creacion de perfiles
    tensorboard = TensorBoard("logs/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S"), profile_batch = 0)
    # Guarda el modelo despues de cada epoca
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(x=imgs_train, y=imgs_mask_train,
              batch_size=BATCH_SIZE,
              epochs=NB_EPOCH,
              verbose=2,
              validation_data=(imgs_validation,imgs_mask_validation),
              callbacks=[model_checkpoint, tensorboard])


    #model.fit_generator(data_generator_keras(x=imgs_train, y=imgs_mask_train, batch_size=BATCH_SIZE, full_dataset=imgs, flip=True),
                #epochs=NB_EPOCH,
                #steps_per_epoch=4, #human dice que = len(ids_trainsplit) / batch_size.
                #verbose=2,
                #callbacks=[model_checkpoint, tensorboard],
                #validation_data=(imgs_validation,imgs_mask_validation))

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
        imsave(os.path.join(pred_dir, str(image_id) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_pred.png'), image)
# -----------------------------------------

    return imgs_mask_test_evaluate


def train_and_predict():
    
    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)
    imgs, imgs_mask = load_data() # float64
    # min-max de imgs: -3.01762446888. - 29.2902694143.
    # min-max de imgs mask: 0.0. - 1.0.

    print('-'*30)
    print('Start Cross Validation...')
    print('-'*30)
    
    kf = KFold(n_splits=len(imgs))
    # Para cada una de las 11 iteraciones, dame los indices de train y test
    CONT_CROSSVALIDATION = 0
    for train_index, test_index in kf.split(imgs): 
        CONT_CROSSVALIDATION = CONT_CROSSVALIDATION + 1
        # iteracion 1: train_index = [1 2 3 4 5 6 7 8 9 10] y test_index = [0]
        #Dentro del bucle hay varios epochs. ??????
        #Devuelve el valor de perdida y los valores de metrica para el modelo en modo de prueba.
        imgs_mask_test_evaluate = model_evaluate(imgs, imgs_mask, train_index, test_index) #[0.4318084716796875, 0.022709527984261513]
        list_predict.append(imgs_mask_test_evaluate[1])

        print('La imagen de test es la: ', test_index)
        print('Model Evaluate: categorical_crossentropy = %f, dice_coef = %f' % (imgs_mask_test_evaluate[0],imgs_mask_test_evaluate[1]))
        print('Lista de las 11 imgs test: {0}.'.format(list_predict))

        break # Solo una iteracion para probar

    result = np.mean(list_predict)
    print('Resultado final: {0}.'.format(result))



if __name__ == '__main__':
    train_and_predict()
