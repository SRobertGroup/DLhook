import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import CustomObjectScope

class unet_prediction():
    def __init__(self):
        def read_image(path):
            x = cv2.imread(path, 1)
            x = np.array(x)
            x=x/255
            return x

        def mask_parse(mask):
            mask = np.squeeze(mask)
            mask = [mask, mask, mask]
            mask = np.transpose(mask, (1, 2, 0))
            return mask

        def conv_block(x, num_filters):
            x = Conv2D(num_filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            x = Conv2D(num_filters, (3, 3), padding="same")(x)
            x = BatchNormalization()(x)
            x = Activation("relu")(x)

            return x

        def build_model():
            size = 1024
            num_filters = [32, 48, 64, 128]
            inputs = Input((size, size, 3))
            skip_x = []
            x = inputs

            ## Encoder
            for f in num_filters:
                x = conv_block(x, f)
                skip_x.append(x)
                x = MaxPool2D((2, 2))(x)

            ## Bridge
            x = conv_block(x, num_filters[-1])

            num_filters.reverse()
            skip_x.reverse()

            ## Decoder
            for i, f in enumerate(num_filters):
                x = UpSampling2D((2, 2))(x)
                xs = skip_x[i]
                x = Concatenate()([x, xs])
                x = conv_block(x, f)

            ## Output
            x = Conv2D(1, (1, 1), padding="same")(x)
            x = Activation("sigmoid")(x)

            return Model(inputs, x)

        model=build_model()
        opt = tf.keras.optimizers.Adam()
        model.compile(loss="binary_crossentropy", optimizer=opt)

        weight_path='weights/'
        images_path='data/images/'
        prediction_path='data/predict/'
        for weights in os.listdir(weight_path):
            model.load_weights(weight_path+weights)
            print(weights)
            if "cotyledon" in weights:
                id_n='1'
            if "hypocotyl" in weights:
                id_n='2'

            for image_name in os.listdir(images_path):
                img_x=read_image(images_path+image_name)
                y_pred = model.predict(np.expand_dims(img_x, axis=0))[0] > 0.5
                image = mask_parse(y_pred) * 255.0

                filename=image_name[:-4]+'-'+id_n+'.png'

                cv2.imwrite(prediction_path+filename, image)

