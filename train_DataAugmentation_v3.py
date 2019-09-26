from __future__ import print_function
import os
import itertools

import random
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from libtiff import TIFFimage
from time import time
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


NB_EPOCH = 6
BATCH_SIZE = 32
image_rows = 256
image_cols = 256
list_predict = []
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


def model_evaluate(imgs, imgs_mask, train_index, test_index):
    
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
    
    # TensorBoard, representacion de imagenes y mascaras de entrenamiento
    images_tb = np.reshape(imgs[0:11], (-1, 256, 256, 1))
    masks_tb = np.reshape(imgs_mask[0:11], (-1, 256, 256, 2))
    logdir = "logs/images/{}".format("images.{}".format(int(time())))
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    # Using the file writer, log the reshaped image.
    with file_writer.as_default():
        tf.summary.image("11 Train Images", images_tb, max_outputs=11, step=0)  
        tf.summary.image("11 Mask Images", masks_tb, max_outputs=11, step=0)   
    

    model = get_unet()
    tensorboard = TensorBoard(log_dir="logs/fit/{}".format("fit.{}".format(int(time()))), histogram_freq=1, batch_size=32, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    
    # Guarda el modelo despues de cada epoca
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit_generator(data_generator_keras(x=imgs_train, y=imgs_mask_train, batch_size=BATCH_SIZE, full_dataset=imgs, flip=True),
                epochs=NB_EPOCH,
                steps_per_epoch=4,
                verbose=2,
                callbacks=[model_checkpoint, tensorboard],
                #callbacks=[model_checkpoint], 
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
    for train_index, test_index in kf.split(imgs): 

        # iteracion 1: train_index = [1 2 3 4 5 6 7 8 9 10] y test_index = [0]
        #Dentro del bucle hay varios epochs. ??????
        #Devuelve el valor de perdida y los valores de metrica para el modelo en modo de prueba.
        imgs_mask_test_evaluate = model_evaluate(imgs, imgs_mask, train_index, test_index) #[0.4318084716796875, 0.022709527984261513]
        list_predict.append(imgs_mask_test_evaluate[1])

        print('Model Evaluate: categorical_crossentropy = %f, dice_coef = %f' % (imgs_mask_test_evaluate[0],imgs_mask_test_evaluate[1]))
        print('Lista de las 11 imgs test: {0}.'.format(list_predict))

        break # Solo una iteracion para probar

    result = np.mean(list_predict)
    print('Resultado final: {0}.'.format(result))



if __name__ == '__main__':
    train_and_predict()
