import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# import matplotlib
# matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

class MoeModelV2Good:
    def __init__(self, weight_path=None, class_names=None):
        self.weight_path = weight_path
        self.model = None
        self.class_names = class_names
        self.input_shape = (32, 32, 3)
        self.num_classes = 46 if class_names is None else len(class_names)
        self.num_experts = 4
        self.initial_learning_rate = 0.001
        self.min_learning_rate = 0.00001
        
        # Create the model architecture
        self.create_model(
            input_shape=self.input_shape, 
            num_classes=self.num_classes, 
            num_experts=self.num_experts
        )
        
        # Load weights if path is provided
        if self.weight_path and os.path.exists(self.weight_path):
            self.load_weights()

    def create_model(self, input_shape, num_classes, num_experts):
        input_layer = Input(shape=input_shape)
        
        # Create diverse experts
        expert1 = self.create_efficientnet_b0_expert(input_shape, num_classes, name_prefix="expert1")
        expert2 = self.create_efficientnet_b3_expert(input_shape, num_classes, name_prefix="expert2")
        expert3 = self.create_efficientnet_b4_expert(input_shape, num_classes, name_prefix="expert3")
        expert4 = self.create_custom_cnn_expert(input_shape, num_classes, name_prefix="expert4")

        experts = [ expert1, expert2, expert3, expert4 ]
        
        # Enhanced gating network
        gating_network = self.create_enhanced_gating_network(input_shape, num_experts)
        gate_outputs = gating_network(input_layer)
        expert_outputs = [expert(input_layer) for expert in experts]
        
        # Weighted combination of expert outputs
        moe_output = layers.Lambda(lambda inputs: tf.reduce_sum(
            tf.reshape(inputs[0], [-1, 1, num_experts]) * tf.stack(inputs[1], axis=2),
            axis=2
        ))([gate_outputs, expert_outputs])
        
        self.model = models.Model(inputs=input_layer, outputs=moe_output, name="enhanced_moe_model")
        # return self.model
    
    def load_and_preprocess_data(self, dataset_path, image_size=None, batch_size=32, 
                                validation_split=0.2, split='training', seed=42):
        """
        Improved data loading function that handles class mapping properly
        """
        if image_size is None:
            image_size = self.input_shape[:2]
            
        if split == 'training':
            folder_name = 'Train'
        elif split == 'validation':
            folder_name = 'Train'
        elif split == 'test':
            folder_name = 'Test'
        else:
            raise ValueError(f"Invalid split argument: {split}. Must be 'training', 'validation', or 'test'.")

        folder_path = os.path.join(dataset_path, folder_name)
        print(f"Loading {split} data from: {folder_path}")
        
        # Check if folder exists
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        # Data augmentation for training set only
        if split == 'training':
            data_gen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=6,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                shear_range=0.1,
                brightness_range=(0.9, 1.1),
                horizontal_flip=False,  # No horizontal flip for text characters
                validation_split=validation_split
            )
        else:
            # Just rescaling for validation and test sets
            data_gen = ImageDataGenerator(
                rescale=1./255,
                validation_split=validation_split if split == 'validation' else None
            )
        
        # Load the dataset
        subset = None
        if split == 'training':
            subset = 'training'
        elif split == 'validation':
            subset = 'validation'
        # For test, subset remains None
        
        dataset = data_gen.flow_from_directory(
            directory=folder_path,
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            shuffle=True if split == 'training' else False,
            seed=seed,
            subset=subset
        )
        
        # Update class names from the data generator
        if self.class_names is None and hasattr(dataset, 'class_indices'):
            # Sort by index to get correct order
            self.class_names = [k for k, v in sorted(dataset.class_indices.items(), key=lambda x: x[1])]
            self.num_classes = len(self.class_names)
            print(f"Loaded class names: {self.class_names}")
        
        print(f"Found {dataset.samples} images belonging to {dataset.num_classes} classes")
        print(f"Class indices: {dataset.class_indices}")
        
        return dataset
    
    def predict(self, image_path):
        """
        Make predictions on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with predicted class and confidence score
        """


        print("Inside predict function!")
        print(image_path)
        # Load and preprocess the image
        img = Image.open(image_path).resize((32, 32)).convert("RGB")
        img_array = np.array(img)

        # # Save the image instead of displaying it
        # plt.figure(figsize=(6, 6))
        # plt.imshow(img_array)
        # plt.title("Input Image")
        # plt.axis('on')
        # plt.savefig('input_visualization.png')
        # plt.close()  # Important to close the figure

        img_array = img_array / 255.0  # Normalize pixel values
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension
        
        # Make prediction
        predictions = self.model.predict(img_array)
        scores = tf.nn.softmax(predictions)
        
        # Get prediction results
        predicted_class_idx = np.argmax(scores)
        confidence = float(scores[0][predicted_class_idx])
        
        if self.class_names and len(self.class_names) > predicted_class_idx:
            predicted_label = self.class_names[predicted_class_idx]
        else:
            predicted_label = f"Class {predicted_class_idx}"
        
        return {
            'class': predicted_label,
            'confidence': confidence
        }


    def evaluate_on_test_data(self, dataset_path, batch_size=32):
        """
        Evaluate model on test dataset using the proper data loading function
        """
        # Load test data
        test_dataset = self.load_and_preprocess_data(
            dataset_path=dataset_path,
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            split='test',
            seed=42
        )
        
        print(f"Evaluating on {test_dataset.samples} test images...")
        
        # Get predictions
        predictions = self.model.predict(test_dataset, verbose=1)
        predicted_classes = np.argmax(predictions, axis=1)
        
        print("prediction is already called")

        print(predicted_classes)
        # Get true labels
        true_classes = test_dataset.classes
        
        # Calculate accuracy
        accuracy = np.mean(predicted_classes == true_classes)
        
        # Create class name mappings
        class_names = list(test_dataset.class_indices.keys())
        
        # Convert indices to class names
        predicted_labels = [class_names[i] for i in predicted_classes]
        true_labels = [class_names[i] for i in true_classes]
        
        # Create results dictionary
        results = {
            'predictions': predicted_labels,
            'true_labels': true_labels,
            'accuracy': accuracy,
            'predicted_indices': predicted_classes,
            'true_indices': true_classes
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Save results
        results_df = pd.DataFrame({
            'True Label': true_labels,
            'Predicted Label': predicted_labels,
            'True Index': true_classes,
            'Predicted Index': predicted_classes
        })
        results_df.to_csv('test_predictions_improved.csv', index=False)
        print("Results saved to test_predictions_improved.csv")
        
        return results
    

# Helper function to get class names
def get_class_names():
    class_names = [
        'character_1_ka',
        'character_2_kha',
        'character_3_ga',
        'character_4_gha',
        'character_5_kna',
        'character_6_cha',
        'character_7_chha',
        'character_8_ja',
        'character_9_jha',
        'character_10_yna',
        'character_11_taamatar',
        'character_12_thaa',
        'character_13_daa',
        'character_14_dhaa',
        'character_15_adna',
        'character_16_tabala',
        'character_17_tha',
        'character_18_da',
        'character_19_dha',
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
        'character_30_motosaw',
        'character_31_petchiryakha',
        'character_32_patalosaw',
        'character_33_ha',
        'character_34_chhya',
        'character_35_tra',
        'character_36_gya',
        'digit_0',
        'digit_1',
        'digit_2',
        'digit_3',
        'digit_4',
        'digit_5',
        'digit_6',
        'digit_7',
        'digit_8',
        'digit_9'
    ]
    return class_names

# Example of how to use the model
if __name__ == "__main__":
    #  99.26%
    weights_path = os.path.join(
                os.getcwd(), 
                "../trainModel", 
                "model_checkpoints", 
                "MoeModelV2_moe_optimized_model-3-expert-20-0.9737.weights.h5"
            )

    test_dir = os.path.join(os.getcwd(), "../data", "DevanagariHandwrittenCharacterDataset", "SampleRandom")

    model1 = MoeModelV2Good(weight_path=weights_path, class_names=get_class_names())
    
    # Run prediction on all test images
    # test_results = model1.predictTest(test_dir, batch_size=3)
    test_results = model1.evaluate_on_test_data(test_dir, batch_size=1)