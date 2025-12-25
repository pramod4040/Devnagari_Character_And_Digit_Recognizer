import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image, ImageOps
from .CreateEnsModel import getEfficientModel, get_class_names
from .AbstractModel import AbstractDigitModel


class EnsembleModelCharacter(AbstractDigitModel):
    def __init__(self, weight_path, class_names):
        self.weight_path = weight_path
        self.model = None
        self.class_names = class_names

        self.create_model()
        self.load_weights()

    def create_model(self):
        self.model = getEfficientModel()
    
    def load_weights(self):
        self.model.load_weights(self.weight_path)

    def predict(self, imagePath):
        imgpath = imagePath
        img = Image.open(imgpath).resize((32,32)).convert("RGB")

        # print(img.size)
        # img = ImageOps.invert(img)

        imgNumpy = np.array(img)
        imgNumpy = tf.expand_dims(imgNumpy, 0)

        print(imgNumpy.shape)
        
        predictions = self.model.predict(imgNumpy)
        scores = tf.nn.softmax(predictions)

        print(scores)

        predictedLabel = self.class_names[np.argmax(scores)]
        confidence = np.argmax(scores)
        print(predictedLabel)

        return {
            'class': predictedLabel,
            'confidence': float(confidence)
        }
    
    # def predict(self, image):
    #     predictions = self.model.predict(image)
    #     predicted_class = np.argmax(predictions[0])
    #     confidence = predictions[0][predicted_class]
    #     return {
    #         'class': self.class_names[predicted_class],
    #         'confidence': float(confidence)
    #     }



def get_class_names():
    class_names = ['character_10_yna',
    'character_11_taamatar',
    'character_12_thaa',
    'character_13_daa',
    'character_14_dhaa',
    'character_15_adna',
    'character_16_tabala',
    'character_17_tha',
    'character_18_da',
    'character_19_dha',
    'character_1_ka',
    'character_20_na',
    'character_21_pa',
    'character_22_pha',
    'character_23_ba',
    'character_24_bha',
    'character_25_ma',
    'character_26_yaw',
    'character_27_ra',
    'character_28_la',
    'character_29_waw',
    'character_2_kha',
    'character_30_motosaw',
    'character_31_petchiryakha',
    'character_32_patalosaw',
    'character_33_ha',
    'character_34_chhya',
    'character_35_tra',
    'character_36_gya',
    'character_3_ga',
    'character_4_gha',
    'character_5_kna',
    'character_6_cha',
    'character_7_chha',
    'character_8_ja',
    'character_9_jha',
    'digit_0',
    'digit_1',
    'digit_2',
    'digit_3',
    'digit_4',
    'digit_5',
    'digit_6',
    'digit_7',
    'digit_8',
    'digit_9']

    return class_names