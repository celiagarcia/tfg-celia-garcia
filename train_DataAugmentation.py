from __future__ import print_function
#
# Celia García Fernández
# train_DataAugmentation.py para entrenar el modelo con aumento de datos, y evaluarlo.
#
# https://academic.oup.com/bioinformatics/article/30/11/1609/283435
# https://www.nature.com/articles/nmeth.4473
#
#


import os
import itertools
import random
from sklearn.model_selection import KFold
from time import time
from datetime import datetime
from skimage.transform import resize
from skimage.io import imsave
from skimage import exposure
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
from keras.models import Model
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau, CSVLogger

from keras import backend as K
from keras.utils.np_utils import to_categorical 
from keras.preprocessing.image import ImageDataGenerator
from data import load_data_1, load_data_2, load_data_mix
from data_challenge import load_imgs_challenge

from model_unet import get_unet
import tensorflow as tf


NB_EPOCH = 1
BATCH_SIZE = 15
image_rows = 256
image_cols = 256
list_predict_dice = []
list_predict_loss = []
#NUM_VAL= 3
CONT_CROSSVALIDATION = 0

filename_csvlogger = '02_conDA/700epoch/700epoch.csv'
logdir_scalar = os.path.join ("logs","scalar" , "scalar_700epoch")
logdir_imgs = os.path.join ("logs","images" , "images_700epoch")


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)






def deform_pixel(X, Y):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    alpha=15 #factor de escala de deformacion
    sigma=3 #factor de suavizado
    random_state=None
    # img : (256, 256, 2)
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = X.shape
    shapemask = Y.shape

    Y_1 = np.reshape(Y[:,:,1], (256, 256, 1))
    Y_0 = np.reshape(Y[:,:,0], (256, 256, 1))

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="nearest", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="nearest", cval=0) * alpha
    #dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)

    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

    distored_image = map_coordinates(X, indices, order=3, mode='reflect').reshape(shape)
    distored_mask1 = map_coordinates(Y_1, indices, order=3, mode='reflect').reshape(shape)
    distored_mask0 = map_coordinates(Y_0, indices, order=3, mode='reflect').reshape(shape)

    distored_mask = np.concatenate([distored_mask1, distored_mask0], axis=2) #256,256,2
    #imsave(os.path.join('DA_test/', datetime.now().strftime("%Y%m%d-%H%M%S") + '_DAimgtest.png'), distored_image)
    #masksaved = (distored_mask[:, :, 1] * 255.).astype(np.uint8)
    #imsave(os.path.join('DA_test/', datetime.now().strftime("%Y%m%d-%H%M%S") + '_DAmasktest1.png'), distored_mask[:,:,1])
    #imsave(os.path.join('DA_test/', datetime.now().strftime("%Y%m%d-%H%M%S") + '_DAmasktest0.png'), distored_mask[:,:,0])
    return distored_image, distored_mask




def elastic_transform(img):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    alpha=991 #factor de escala de deformacion
    sigma=8 #factor de suavizado
    random_state=None
    # img : (256, 256, 2)

    if random_state is None:
        random_state = np.random.RandomState(None)
    shape = img.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="nearest", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="nearest", cval=0) * alpha
    #dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = np.zeros_like(dx)


    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
    indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))
    distored_image = map_coordinates(img, indices, order=3, mode='reflect').reshape(shape)
    image_saved = (distored_image[:, :, 1] * 255.).astype(np.uint8)
    imsave(os.path.join('DA_def_elas_mask/', datetime.now().strftime("%Y%m%d-%H%M%S") + '_DA.png'), image_saved)

    return distored_image.reshape(img.shape)


def equal_hist(img):
    # Estiramiento de contraste
    # img : (256, 256, 1)
    p2, p98 = np.percentile(img, (random.randint(0, 10), random.randint(90,100)))
    img_rescale = exposure.rescale_intensity(img, in_range=(p2, p98))
    # img_rescale : (256, 256, 1)
    #le añado una deformacion elastica a la imagen
    #img_rescale_t = elastic_transform(img_rescale, 991, 8)
    return img_rescale




def data_generator_keras(x, y, batch_size, full_dataset, flip):
    # IMGS
    datagen_x = ImageDataGenerator(
        # Estandarizar los valores de pixeles en todo el conjunto de datos. Esto se denomina estandarizacion de caracteristicas.
        featurewise_center=True,    # Establece la media de entrada en 0 del conjunto de datos.
        featurewise_std_normalization=True, # Divide las entradas por la desviacion tipica del conjunto de datos.
        # --zca_epsilon,
        # zca_whitening = True,   # Menos redundancia en la imagen, resaltar mejor las estructuras y caracteristicas de la imagen para el algoritmo de aprendizaje.
        rotation_range=60,        # Rango de grados para rotaciones aleatorias
        width_shift_range=0.1,    # Desplazamientos aleatorios horizontales
        height_shift_range=0.1,   # Desplazamientos aleatorios horizontales
        shear_range=0.1,          # Angulo de corte en sentido antihorario en radianes
        zoom_range=0.1,           # Rango para zoom aleatorio. Float o [lower, upper] = [1-zoom_range, 1+zoom_range]
        horizontal_flip=flip,       # Voltea las entradas horizontalmente de forma aleatoria      
        vertical_flip=flip,         # Voltea las entradas verticalmente de forma aleatoria 
        fill_mode='reflect',#'constant',        # Los puntos fuera de los limites de la entrada se rellenan segun el modo. "nearest":  aaaaaaaa|abcd|dddddddd
        preprocessing_function=equal_hist)

    # MASK
    datagen_y = ImageDataGenerator(
        rotation_range=60,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=flip,
        vertical_flip=flip,
        fill_mode='reflect')

    if full_dataset is not None:
        datagen_x.fit(full_dataset) # full_dataset para calcular la media de todas las imagenes.

    for batch in itertools.zip_longest(
        datagen_x.flow(x, batch_size=batch_size, seed=1337), #semilla aleatoria para que sea la misma transformacion en mask e imgs.
        #datagen_x.flow(x, batch_size=batch_size, seed=1337, save_to_dir=os.path.join('DA'), save_prefix='imgsDA', save_format='png'),
        datagen_y.flow(y, batch_size=batch_size, seed=1337)
        #datagen_y.flow(y[:,:,:,1].reshape(x.shape), batch_size=batch_size, seed=1337, save_to_dir=os.path.join('DA'), save_prefix='maskDA', save_format='png')
        ):
        yield batch


def model_evaluate(imgs, imgs_mask, train_index, test_index):
    
    print('-'*30)
    print('Creating the subsets of train, validation and test...')
    print('-'*30)
    imgs_fulltrain = imgs[train_index] # (10, 256, 256, 1)
    imgs_mask_fulltrain = imgs_mask[train_index] # (10, 256, 256, 2)

    #index = np.arange(len(imgs_fulltrain)) # [0,1,2...8,9]
    #index_val = np.random.choice(index, NUM_VAL)
    #index_train = np.delete(index, index_val)

    index_train = np.arange(len(imgs_fulltrain)) # [0,1,2...8,9]
    index_val = np.random.choice(index_train, 1)

    index_train = np.setdiff1d(index_train, index_val)
    index_val = np.append(index_val, np.random.choice(index_train, 1))

    index_train = np.setdiff1d(index_train, index_val)
    index_val = np.append(index_val, np.random.choice(index_train, 1))

    index_train = np.setdiff1d(index_train, index_val)
    index_val = np.append(index_val, np.random.choice(index_train, 1))

    index_train = np.setdiff1d(index_train, index_val)
    index_val = np.append(index_val, np.random.choice(index_train, 1))

    index_train = np.setdiff1d(index_train, index_val)
    index_val = np.append(index_val, np.random.choice(index_train, 1))

    index_train = np.setdiff1d(index_train, index_val)


    print('------------- INDEX VAL:', index_val, ' ----------------')
    print('------------- INDEX TRAIN:', index_train, ' ----------------')

    imgs_train = imgs_fulltrain[index_train]
    imgs_mask_train = imgs_mask_fulltrain[index_train]
    imgs_validation = imgs_fulltrain[index_val]
    imgs_mask_validation = imgs_mask_fulltrain[index_val]

    imgs_test = imgs[test_index]
    imgs_mask_test = imgs_mask[test_index]

    model = get_unet()

    # CALLBACKS
    #log_dir= "logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    #log_dir = os.path.join ("logs","epoch100", datetime.now().strftime("%Y%m%d-%H%M%S")) #esta es la buena
    #tensorboard = TensorBoard(log_dir, histogram_freq=1, batch_size=BATCH_SIZE, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    tensorboard = TensorBoard(logdir_scalar + datetime.now().strftime("%Y%m%d-%H%M%S"), profile_batch = 0)
    model_checkpoint = ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True, mode='min') # Guarda el modelo despues de cada epoca
    earlyStopping = EarlyStopping(monitor= 'val_loss', patience = 20, mode = 'min')
    #reduceLROnPlateau = ReduceLROnPlateau(monitor= 'val_loss', patience = 8, mode = 'min', min_delta = 1e-4, factor = 0.1)
    csvLogger = CSVLogger(filename = filename_csvlogger, separator=',', append='False')

    print('-'*30)
    print('Fitting model...')
    print('-'*30)

    model.fit(x=imgs_train, y=imgs_mask_train,
                batch_size=BATCH_SIZE,
                epochs=NB_EPOCH,
                verbose=2,
                callbacks=[model_checkpoint],
                validation_data=(imgs_validation,imgs_mask_validation),
                steps_per_epoch= None,
                validation_steps=None)

    """#antes de pasarle el datagenerator, voy a aplicarle una deformacion elastica a cada imagen y su mascara.
    for index in index_train: 

        print(index)
        img_dis, mask_dis = deform_pixel(imgs_train[index], imgs_mask_train[index])

        imsave(os.path.join('DA_test/', str(index) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_DAimgtest.png'), img_dis)

        #masksaved = (distored_mask[:, :, 1] * 255.).astype(np.uint8)
        imsave(os.path.join('DA_test/', str(index) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_DAmasktest1.png'), mask_dis[:,:,1])
        imsave(os.path.join('DA_test/', str(index) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_DAmasktest0.png'), mask_dis[:,:,0])

    model.fit(data_generator_keras(x=imgs_train, y=imgs_mask_train, batch_size=BATCH_SIZE, full_dataset=imgs, flip=True),
                epochs=NB_EPOCH,
                steps_per_epoch=2, #steps_per_epoch=len(x_train) / batch_size
                verbose=2,
                callbacks=[model_checkpoint, tensorboard, csvLogger],
                validation_data=(imgs_validation,imgs_mask_validation))# las imagenes de validacion no las transformo"""


    print('-'*30)
    print('Loading saved weights...')
    print('-'*30)
    model.load_weights('weights.h5')
    print('-'*30)
    print('Evaluating masks on test data...')
    print('-'*30)

    # Devuelve el valor de perdida y los valores de metrica para el modelo en modo de prueba.
    imgs_mask_test_evaluate = model.evaluate(imgs_test, imgs_mask_test, verbose=1)

    print('-' * 30)
    print('Saving predicted masks to files...')
    print('-' * 30)

    # Con el modelo entrenado, devuelve la mask con la imagen que le metas.
    imgs_test = imgs[test_index]
    imgs_mask_test_predict = model.predict(imgs_test, verbose=1)
    np.save('imgs_mask_test_predict_mix.npy', imgs_mask_test_predict) # (1, 256, 256, 2)


    pred_dir = 'preds'
    if not os.path.exists(pred_dir):
        os.mkdir(pred_dir)
    for image, image_id in zip(imgs_mask_test_predict, test_index):
        image = (image[:, :, 1] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_pred.png'), image)
        #Esto no vale si el array de test es mayor de 1
        imagemask_ori = (imgs_mask_test[0][:, :, 1] * 255.).astype(np.uint8)
        imsave(os.path.join(pred_dir, str(image_id) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_orig.png'), imagemask_ori)


    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir_imgs + datetime.now().strftime("%Y%m%d-%H%M%S"))
    # Using the file writer, log the reshaped image.
    with file_writer.as_default():
        tf.summary.image(str(test_index) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_pred', imgs_mask_test_predict, step=0)
        tf.summary.image(str(test_index) + '_' + datetime.now().strftime("%Y%m%d-%H%M%S") + '_orig', imgs_test, step=0)  


    return imgs_mask_test_evaluate






def train_and_predict():
    
    print('-'*30)
    print('Loading and preprocessing data...')
    print('-'*30)
    imgs2, imgs_mask2 = load_data_2() 
    imgs1, imgs_mask1 = load_data_1() # float64
    imgs, imgs_mask = load_data_mix() # float64
    # min-max de imgs: -3.01762446888. - 29.2902694143.
    # min-max de imgs mask: 0.0. - 1.0.

    imgs_challenge_1, imgs_challenge_2 = load_imgs_challenge() # float64

    print('-'*30)
    print('Start Cross Validation...')
    print('-'*30)
    
    kf = KFold(n_splits=len(imgs))
    #kf = KFold(n_splits=13)
    # Para cada una de las 11 iteraciones, dame los indices de train y test
    CONT_CROSSVALIDATION = 0
    for train_index, test_index in kf.split(imgs): 
        CONT_CROSSVALIDATION = CONT_CROSSVALIDATION + 1

        #if CONT_CROSSVALIDATION == 12:
            #break # Solo una iteracion para probar sin validacion cruzada. 

        #print('Estoy en la iteracion %f de cross validation. Con la imagen de test de index %f.' % (CONT_CROSSVALIDATION, test_index))
        #Devuelve el valor de perdida y los valores de metrica para el modelo en modo de prueba.
        imgs_mask_test_evaluate = model_evaluate(imgs, imgs_mask, train_index, test_index) #[0.4318084716796875, 0.022709527984261513]
        list_predict_dice.append(imgs_mask_test_evaluate[1])
        list_predict_loss.append(imgs_mask_test_evaluate[0])

        print('Model Evaluate: categorical_crossentropy = %f, dice_coef = %f' % (imgs_mask_test_evaluate[0],imgs_mask_test_evaluate[1]))
        print('Lista de las 28 imgs test dice: {0}.'.format(list_predict_dice))
        print('Lista de las 28 imgs test loss: {0}.'.format(list_predict_loss))


    result_dice = np.mean(list_predict_dice)
    print('Resultado final dice: {0}.'.format(result_dice))

    result_loss = np.mean(list_predict_loss)
    print('Resultado final loss: {0}.'.format(result_loss))



if __name__ == '__main__':
    train_and_predict()
