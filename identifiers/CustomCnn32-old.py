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

# import matplotlib
# matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class CustomCNNModel:
    def __init__(self, weight_path=None, class_names=None):
        self.weight_path = weight_path
        self.model = None
        self.class_names = class_names
        self.input_shape = (32, 32, 3)
        self.num_classes = 46 if class_names is None else len(class_names)
        self.initial_learning_rate = 0.001
        self.min_learning_rate = 0.00001
        
        # Create the model architecture
        self.create_custom_cnn_expert(
            input_shape=self.input_shape, 
            num_classes=self.num_classes
        )
        
        # Load weights if path is provided
        if self.weight_path and os.path.exists(self.weight_path):
            self.load_weights()

    def create_custom_cnn_expert(self, input_shape, num_classes, name_prefix="custom_cnn_one"):
        input_tensor = Input(shape=input_shape)
        
        # First convolutional block
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolutional block
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolutional block
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Flatten and dense layers
        x = layers.Flatten()(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output_tensor = layers.Dense(num_classes, activation='softmax', name=f'{name_prefix}_output')(x)
        
        # a = models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')
        self.model = models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')
    
    def load_weights(self):
        """Load model weights from the specified path"""
        try:
            # Try loading weights without the by_name parameter
            self.model.load_weights(self.weight_path)
            print(f"Successfully loaded weights from {self.weight_path}")
            
        except Exception as e:
            # If that fails, try alternative approaches
            try:
                # Try load_model if available
                if hasattr(tf.keras.models, 'load_model'):
                    temp_model = tf.keras.models.load_model(self.weight_path)
                    self.model.set_weights(temp_model.get_weights())
                    print(f"Successfully loaded weights via load_model from {self.weight_path}")
                else:
                    print(f"Could not load weights: {e}")
            except Exception as e2:
                print(f"Error loading weights: {e2}")
        
        # Compile the model after loading weights
        optimizer = Adam(learning_rate=self.initial_learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def save_weights(self, filepath):
        """Save model weights to the specified path"""
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")

    def predict(self, image_path):
        """
        Make predictions on an image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with predicted class and confidence score
        """
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
    
    def predict_batch(self, images):
        """
        Make predictions on a batch of images
        
        Args:
            images: Batch of images as numpy array [batch_size, height, width, channels]
            
        Returns:
            Array of dictionaries with predicted classes and confidence scores
        """
        # Ensure images are normalized
        if images.max() > 1.0:
            images = images / 255.0
            
        predictions = self.model.predict(images)
        results = []
        
        for i, pred in enumerate(predictions):
            scores = tf.nn.softmax(pred)
            predicted_class_idx = np.argmax(scores)
            confidence = float(scores[predicted_class_idx])
            
            if self.class_names and len(self.class_names) > predicted_class_idx:
                predicted_label = self.class_names[predicted_class_idx]
            else:
                predicted_label = f"Class {predicted_class_idx}"
            
            results.append({
                'class': predicted_label,
                'confidence': confidence
            })
        
        return results
    
    def compile(self, learning_rate=0.001):
        """Compile the model with specified parameters"""
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("Model compiled successfully.")

    def summary(self):
        """Display model summary"""
        return self.model.summary()
    
    def predictTest(self, test_dir, batch_size=32):
        """
        Processes all images in test directory and evaluates model performance
        
        Args:
            model: Trained model instance
            test_dir: Directory containing test images in class folders
            batch_size: Batch size for prediction
            
        Returns:
            Dictionary with accuracy metrics and prediction results
        """

        results = {'predictions': [], 'true_labels': [], 'accuracy': 0}
        all_images = []
        all_labels = []
        
        # Walk through test directory
        print(f"Loading test images from {test_dir}")
        class_dirs = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
        
        for class_dir in tqdm(class_dirs, desc="Loading classes"):
            class_path = os.path.join(test_dir, class_dir)
            # Skip non-directories
            if not os.path.isdir(class_path):
                continue
                
            # Get actual class name
            true_class = class_dir
            
            # Process each image in this class
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).resize((32, 32)).convert("RGB")
                    img_array = np.array(img) / 255.0  # Normalize
                    
                    # Add to batches
                    all_images.append(img_array)
                    all_labels.append(true_class)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy arrays
        all_images = np.array(all_images)
        
        # Process in batches
        total_images = len(all_images)
        print(f"Total test images: {total_images}")
        
        # Make predictions in batches
        all_predictions = []
        for i in range(0, total_images, batch_size):
            batch_images = all_images[i:i+batch_size]
            batch_preds = self.predict_batch(batch_images)
            all_predictions.extend(batch_preds)
        
        # Calculate accuracy
        correct = 0
        for i, pred in enumerate(all_predictions):
            results['predictions'].append(pred['class'])
            results['true_labels'].append(all_labels[i])
            
            if pred['class'] == all_labels[i]:
                correct += 1
        
        results['accuracy'] = correct / total_images if total_images > 0 else 0
        
        # Create a dataframe for analysis
        results_df = pd.DataFrame({
            'True Label': results['true_labels'],
            'Predicted Label': results['predictions']
        })
        
        # Save results to CSV
        results_df.to_csv('test_predictions.csv', index=False)
        
        print(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}")
        print(f"Results saved to test_predictions.csv")
        
        return results
    
    def evaluate_metrics(self, predictions, true_labels, class_names=None):
        """
        Computes and prints evaluation metrics and plots confusion matrix.

        Args:
            predictions: List of predicted class labels
            true_labels: List of true class labels
            class_names: Optional list of class names to label axes

        Returns:
            Dictionary with confusion matrix and macro metrics
        """
        print("\nClassification Report:")
        report = classification_report(true_labels, predictions, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        # Default to sorted set of labels if class_names not provided
        if class_names is None:
            class_names = sorted(list(set(true_labels + predictions)))
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        plt.show()

        macro_metrics = {
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1_score': report['macro avg']['f1-score'],
            # 'confusion_matrix': cm.tolist()
        }
        
        print("\nMacro Metrics:")
        for k, v in macro_metrics.items():
            if k != "confusion_matrix":
                print(f"{k}: {v:.4f}")
        
        return macro_metrics



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
    weight_path = os.path.join(
        os.getcwd(), 
        "../trainModel", 
        "model_checkpoints", 
        "CustomCNNOnlyV2-32-14-0.9557.weights.h5"
    )

    test_dir = os.path.join(os.getcwd(), "../data", "DevanagariHandwrittenCharacterDataset", "SmallDataSet", "Test")

    model1 = CustomCNNModel(weight_path=weight_path, class_names=get_class_names())
    
    # # Run prediction on all test images
    test_results = model1.predictTest(test_dir, batch_size=20)

    results = model1.predictTest(test_dir=test_dir)
    metrics = model1.evaluate_metrics(results['predictions'], results['true_labels'])
    print(metrics)
    