from tensorflow.keras import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense

class DevnagariCharacter:
    def __init__(self, weightPath):
        self.weightPath = weightPath

    def create_model(self):
        model = Sequential()

        # Block 1
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
        model.add(Conv2D(8, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Block 2
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Block 3
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Block 4
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Block 5
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
        # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        # Fully connected layers
        model.add(Flatten())
        model.add(Dense(1048, activation='relu'))
        model.add(Dense(1048, activation='relu'))
        model.add(Dense(36, activation='softmax'))  # Assumes 36 classes

        return model
    
    def load_trained_model(self):
        model = self.create_model()
        model.load_weights(self.weights_path)
        return model
    