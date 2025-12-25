import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

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
        self.history = None
        
        # Create the model architecture
        self.create_custom_cnn_expert(
            input_shape=self.input_shape, 
            num_classes=self.num_classes
        )
        print(self.weight_path)
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

    def load_data_from_directory(self, data_dir, test_size=0.2, validation_size=0.1):
        """
        Load images and labels from directory structure
        
        Args:
            data_dir: Directory containing class folders with images
            test_size: Fraction of data to use for testing
            validation_size: Fraction of training data to use for validation
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        print(f"Loading data from {data_dir}")
        
        images = []
        labels = []
        class_names = []
        
        # Get class directories
        class_dirs = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
        for class_idx, class_dir in enumerate(tqdm(class_dirs, desc="Loading classes")):
            class_path = os.path.join(data_dir, class_dir)
            class_names.append(class_dir)
            
            # Process images in this class
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Load and preprocess image
                    img = Image.open(img_path).resize(self.input_shape[:2]).convert("RGB")
                    img_array = np.array(img) / 255.0  # Normalize
                    
                    images.append(img_array)
                    labels.append(class_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X = np.array(images)
        y = np.array(labels)
        
        # Update class names if not provided
        if self.class_names is None:
            self.class_names = class_names
            self.num_classes = len(class_names)
        
        # Convert labels to categorical
        y_categorical = to_categorical(y, num_classes=self.num_classes)
        
        # Split data
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y_categorical, test_size=test_size, random_state=42, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=validation_size, random_state=42, stratify=np.argmax(y_temp, axis=1)
        )
        
        print(f"Data loaded successfully:")
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of classes: {self.num_classes}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test

    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32, augment=True):
        """
        Create data generators for training and validation
        
        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            batch_size: Batch size for training
            augment: Whether to apply data augmentation
            
        Returns:
            Tuple of (train_generator, val_generator)
        """
        if augment:
            # Data augmentation for training
            train_datagen = ImageDataGenerator(
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,  # Typically not good for text/characters
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator()
        
        # No augmentation for validation
        val_datagen = ImageDataGenerator()
        
        train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
        val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)
        
        return train_generator, val_generator

    def train(self, data_dir=None, X_train=None, y_train=None, X_val=None, y_val=None, 
              epochs=10, batch_size=32, learning_rate=0.001, 
              save_best_weights=True, weights_save_path="model_weights.h5",
              early_stopping_patience=10, lr_reduce_patience=5, augment_data=True):
        """
        Train the model
        
        Args:
            data_dir: Directory containing training data (alternative to providing X_train, y_train directly)
            X_train, y_train, X_val, y_val: Training and validation data
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Initial learning rate
            save_best_weights: Whether to save best weights during training
            weights_save_path: Path to save the best weights
            early_stopping_patience: Patience for early stopping
            lr_reduce_patience: Patience for learning rate reduction
            augment_data: Whether to apply data augmentation
            
        Returns:
            Training history
        """
        # Load data if directory is provided
        if data_dir is not None:
            X_train, X_val, _, y_train, y_val, _ = self.load_data_from_directory(data_dir)
        
        if X_train is None or y_train is None:
            raise ValueError("Either provide data_dir or X_train/y_train")
        
        # Compile model
        self.compile(learning_rate=learning_rate)
        
        # Create data generators
        train_generator, val_generator = self.create_data_generators(
            X_train, y_train, X_val, y_val, batch_size, augment_data
        )
        
        # Callbacks
        callbacks = []
        
        if save_best_weights:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(weights_save_path), exist_ok=True)
            
            checkpoint = ModelCheckpoint(
                weights_save_path,
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(checkpoint)
        
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=early_stopping_patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=lr_reduce_patience,
            min_lr=self.min_learning_rate,
            verbose=1
        )
        callbacks.append(lr_scheduler)
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // batch_size
        validation_steps = len(X_val) // batch_size
        
        print(f"Starting training...")
        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history

    def plot_training_history(self, save_path="training_history.png"):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.show()
        print(f"Training history plot saved to {save_path}")
    
    def load_weights(self):
        """Load model weights from the specified path"""
        try:
            # Try loading weights without the by_name parameter
            self.model.load_weights(self.weight_path)
            print(f"Successfully loaded weights from {self.weight_path}")
            
        except Exception as e:
            # If that fails, try alternative approaches
            print(f"❌ Error loading weights: {e}")
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
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")

    def save_model(self, filepath):
        """Save entire model to the specified path"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

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
    

    def predictTest_fixed(self, test_dir, batch_size=32):
        """
        Fixed version with better class name handling and debugging
        """
        results = {'predictions': [], 'true_labels': [], 'accuracy': 0}
        all_images = []
        all_labels = []
        
        print(f"Expected class names: {self.class_names}")
        print(f"Loading test images from {test_dir}")
        
        # Get class directories and sort them
        class_dirs = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
        print(f"Found test directories: {class_dirs}")
        
        # Create mapping from directory names to expected class names
        dir_to_class_mapping = {}
        for class_dir in class_dirs:
            # Try to find matching class name
            matching_class = None
            for expected_class in self.class_names:
                if class_dir in expected_class or expected_class in class_dir:
                    matching_class = expected_class
                    break
            
            if matching_class:
                dir_to_class_mapping[class_dir] = matching_class
            else:
                # Fallback: use directory name as is
                dir_to_class_mapping[class_dir] = class_dir
                print(f"Warning: No matching class found for directory '{class_dir}'")
        
        print(f"Directory to class mapping: {dir_to_class_mapping}")
        
        for class_dir in tqdm(class_dirs, desc="Loading classes"):
            class_path = os.path.join(test_dir, class_dir)
            
            # Use mapped class name
            true_class = dir_to_class_mapping[class_dir]
            
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_file in image_files:
                img_path = os.path.join(class_path, img_file)
                try:
                    # Use same preprocessing as training
                    img = Image.open(img_path).resize((32, 32)).convert("RGB")
                    img_array = np.array(img, dtype=np.float32)
                    # / 255.0  # Explicit dtype
                    
                    all_images.append(img_array)
                    all_labels.append(true_class)
                    
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
        
        # Convert to numpy array with explicit dtype
        all_images = np.array(all_images, dtype=np.float32)
        print(f"Final image array shape: {all_images.shape}")
        print(f"Final image array dtype: {all_images.dtype}")
        print(f"Image value range: {all_images.min()} to {all_images.max()}")
        
        # Test a few predictions to debug
        print("\nTesting first few predictions:")
        for i in range(min(3, len(all_images))):
            single_pred = self.model.predict(np.expand_dims(all_images[i], 0))
            pred_class_idx = np.argmax(single_pred)
            confidence = np.max(single_pred)
            pred_class = self.class_names[pred_class_idx] if pred_class_idx < len(self.class_names) else f"Class_{pred_class_idx}"
            print(f"True: {all_labels[i]}, Pred: {pred_class}, Confidence: {confidence:.3f}")
        
        # Continue with batch prediction...
        total_images = len(all_images)
        print(f"Total test images: {total_images}")
        
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
        
        print(f"Evaluation complete. Accuracy: {results['accuracy']:.4f}")
        return results

    def predictTestNew(self, test_dir, batch_size=32):
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

# Example usage scenarios
if __name__ == "__main__":
    # Scenario 1: Train a new model from scratch
    # print("=== Training Example ===")
    # train_data_dir = "../data/DevanagariHandwrittenCharacterDataset/SmallDataSet/Train"
    
    # # Create model instance
    # model = CustomCNNModel(class_names=get_class_names())
    
    # # Train the model
    # history = model.train(
    #     data_dir=train_data_dir,
    #     epochs=6,
    #     batch_size=32,
    #     learning_rate=0.0001,
    #     save_best_weights=True,
    #     weights_save_path="model_checkpoints/CustomCnn-After-mid-best_weights.weights.h5",
    #     early_stopping_patience=6,
    #     augment_data=True
    # )
    
    # # Plot training history
    # model.plot_training_history("training_plots.png")
    
    # # Save the final model
    # model.save_model("saved_models/custom-cnn-final_model.h5")
    
    # print("\n" + "="*50)

    # Load Mode Scenario: 1.5
    # model_path = os.path.join(
    #     os.getcwd(), 
    #     "saved_models", 
    #     "custom-cnn-final_model.h5"
    # )
    # newModel = load_model(model_path)
    # print("model loaded")
    

    # '''
    # Scenario 2: Load pre-trained weights and test
    print("=== Testing with Pre-trained Weights ===")
    weight_path = os.path.join(
        os.getcwd(), 
        "model_checkpoints", 
        "CustomCnn-After-mid-best_weights.weights.h5"
    )

    # weight_path = os.path.join(
    #     os.getcwd(), 
    #     "saved_models", 
    #     "custom-cnn-final_model.h5"
    # )

    test_dir = os.path.join(os.getcwd(), "../data", "DevanagariHandwrittenCharacterDataset", "SmallDataSet", "Test")

    # # Load model with pre-trained weights
    model_pretrained = CustomCNNModel(weight_path=weight_path, class_names=get_class_names())
    
    # # Run prediction on all test images
    test_results = model_pretrained.predictTestNew(test_dir, batch_size=20)
    
    # # Evaluate metrics
    metrics = model_pretrained.evaluate_metrics(test_results['predictions'], test_results['true_labels'])
    print(f"\nFinal Test Accuracy: {test_results['accuracy']:.4f}")
    print(metrics)
    
    print("\n" + "="*50)

    # '''
    
    # Scenario 3: Single image prediction
    # print("=== Single Image Prediction ===")
    # Assuming you have a single test image
    # single_image_path = "path/to/single/test/image.jpg"
    # prediction = model_pretrained.predict(single_image_path)
    # print(f"Predicted class: {prediction['class']}")
    # print(f"Confidence: {prediction['confidence']:.4f}")