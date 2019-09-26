from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras import backend as K

image_rows = 256
image_cols = 256
smooth = 1.

# Se le pasa el mask que ha creado el modelo, y el mask original, para ver como de bueno es el resultado
def dice_coef(y_true, y_pred):
    # El coeficiente dice compara dos planos, por eso cojo la segunda capa de los mask (capa celula)
    y_true = y_true[:,:,:,1]
    y_pred = y_pred[:,:,:,1]
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


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

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3) # (32, 32, 512)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6) # (32, 32, 256)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6) # (32, 32, 256)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3) # (64, 64, 256)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7) # (64, 64, 128)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7) # (64, 64, 128) 

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3) # (128, 128, 128)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8) # (128, 128, 64)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8) # (128, 128, 64)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3) # (256, 256, 64)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9) # (256, 256, 32)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9) # (256, 256, 32)

    # Usar Softmax para tener la distribucion de probabilidad en la direccion de profundidad (capas)
    # La salida es distribucion de probabilidad en dos capas "celula" y "fondo"
    conv10 = Conv2D(2, (1, 1), activation='softmax')(conv9) # (256, 256, 2)

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


if __name__ == '__main__':
    get_unet()

