import os
import tensorflow as tf
from tensorflow.keras.models import model_from_json
import numpy as np
from PIL import Image, ImageOps
from .CreateEnsModel import getEfficientModel, get_class_names
from .AbstractModel import AbstractDigitModel

from tensorflow.keras import layers, models, Input
# from tensorflow.keras.datasets import mnist  # No longer using MNIST
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint  # Import ModelCheckpoint callback


# Mxture of expert
class MoeModelOne(AbstractDigitModel):
    def __init__(self, weight_path, class_names):
        self.weight_path = weight_path
        self.model = None
        self.class_names = class_names
        self.input_shape = (32, 32, 3)
        self.num_classes = 46
        self.num_experts = 2

        self.create_model(input_shape=self.input_shape, num_classes=self.num_classes, num_experts=self.num_experts)
        self.load_weights()

    def create_efficientnet_expert_model(self, input_shape, num_classes, name_prefix="expert"):
        input_tensor = Input(shape=input_shape)
        resized_input = layers.Resizing(224, 224)(input_tensor)
        base_efficientnet = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=resized_input
        )
        for layer in base_efficientnet.layers:
            layer.trainable = False
        x = GlobalAveragePooling2D()(base_efficientnet.output)
        x = Dense(128, activation='relu')(x)
        output_tensor = Dense(num_classes, activation='softmax', name=f'{name_prefix}_output')(x)
        return models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}_efficientnet')
    
    def create_expert_model_1(self, input_shape, num_classes):
        return self.create_efficientnet_expert_model(input_shape, num_classes, name_prefix="expert1")

    def create_expert_model_2(self, input_shape, num_classes):
        return self.create_efficientnet_expert_model(input_shape, num_classes, name_prefix="expert2")
    
    def create_gating_network(self,input_shape, num_experts):
        input_tensor = Input(shape=input_shape)
        x = layers.Flatten()(input_tensor)
        x = layers.Dense(32, activation='relu')(x)
        output_tensor = layers.Dense(num_experts, activation='softmax')(x)
        return models.Model(inputs=input_tensor, outputs=output_tensor)

    def create_model(self, input_shape, num_classes, num_experts):
        input_layer = Input(shape=input_shape)
        expert1 = self.create_expert_model_1(input_shape, num_classes)
        expert2 = self.create_expert_model_2(input_shape, num_classes)
        experts = [expert1, expert2]
        gating_network = self.create_gating_network(input_shape, num_experts)
        gate_outputs = gating_network(input_layer)
        expert_outputs = [expert(input_layer) for expert in experts]

        # Wrap TensorFlow operations in a Lambda layer
        moe_output = layers.Lambda(lambda inputs: tf.reduce_sum(
            tf.reshape(inputs[0], [-1, 1, num_experts]) * tf.stack(inputs[1], axis=2),
            axis=2
        ))([gate_outputs, expert_outputs])  # Pass gate_outputs and expert_outputs as inputs to the Lambda layer

        self.model = models.Model(inputs=input_layer, outputs=moe_output)
    
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