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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json



class CustomCNNModel:
    def __init__(self, weight_path=None, class_names=None, input_shape=(32, 32, 3)):
        self.weight_path = weight_path
        self.model = None
        self.class_names = class_names
        self.input_shape = input_shape
        self.num_classes = 46 if class_names is None else len(class_names)
        self.initial_learning_rate = 0.001
        self.min_learning_rate = 0.00001
        self.history = None
        
        # Create the model architecture
        self.create_custom_cnn_expert(
            input_shape=self.input_shape, 
            num_classes=self.num_classes
        )
        print(f"Model created with input shape: {self.input_shape}, classes: {self.num_classes}")
        
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
        
        self.model = models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')

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

    def train_with_generators(self, dataset_path, epochs=10, batch_size=32, learning_rate=0.001,
                             validation_split=0.2, save_best_weights=True, 
                             weights_save_path="model_checkpoints/best_weights.weights.h5",
                             early_stopping_patience=10, lr_reduce_patience=5, seed=42):
        """
        Train the model using the improved data loading function
        """
        # Load training and validation data
        train_dataset = self.load_and_preprocess_data(
            dataset_path=dataset_path,
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            validation_split=validation_split,
            split='training',
            seed=seed
        )
        
        val_dataset = self.load_and_preprocess_data(
            dataset_path=dataset_path,
            image_size=self.input_shape[:2],
            batch_size=batch_size,
            validation_split=validation_split,
            split='validation',
            seed=seed
        )
        
        # Update model if number of classes changed
        if self.num_classes != train_dataset.num_classes:
            print(f"Updating model for {train_dataset.num_classes} classes")
            self.num_classes = train_dataset.num_classes
            self.create_custom_cnn_expert(
                input_shape=self.input_shape,
                num_classes=self.num_classes
            )
        
        # Compile model
        self.compile(learning_rate=learning_rate)
        
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
        
        print(f"Starting training...")
        print(f"Training samples: {train_dataset.samples}")
        print(f"Validation samples: {val_dataset.samples}")
        print(f"Steps per epoch: {train_dataset.samples // batch_size}")
        print(f"Validation steps: {val_dataset.samples // batch_size}")
        
        # Train the model
        self.history = self.model.fit(
            train_dataset,
            epochs=epochs,
            validation_data=val_dataset,
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history

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

    def plot_training_history(self, save_path="training_history.png"):
        """
        Plot training history
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training history plot saved to {save_path}")

    def load_weights(self):
        """Load model weights from the specified path"""
        try:
            self.model.load_weights(self.weight_path)
            print(f"Successfully loaded weights from {self.weight_path}")
            
        except Exception as e:
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
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save_weights(filepath)
        print(f"Weights saved to {filepath}")

    def save_model(self, filepath):
        """Save entire model to the specified path"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")

    def predict(self, image_path):
        """
        Make predictions on an image
        """
        # Load and preprocess the image
        img = Image.open(image_path).resize(self.input_shape[:2]).convert("RGB")
        img_array = np.array(img) / 255.0  # Normalize pixel values
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
            'confidence': confidence,
            'class_index': int(predicted_class_idx)
        }

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

    def evaluate_metrics(self, predictions, true_labels, class_names=None):
        """
        Computes and prints evaluation metrics and plots confusion matrix.
        """
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions))
        
        report = classification_report(true_labels, predictions, output_dict=True)
        cm = confusion_matrix(true_labels, predictions)
        
        # Default to sorted set of labels if class_names not provided
        if class_names is None:
            class_names = sorted(list(set(true_labels + predictions)))
        
        # Plot confusion matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("confusion_matrix_improved.png", dpi=300, bbox_inches='tight')
        plt.show()

        macro_metrics = {
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1_score': report['macro avg']['f1-score'],
            'accuracy': report['accuracy']
        }
        
        print("\nMacro Metrics:")
        for k, v in macro_metrics.items():
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

# Example usage with improved functionality
if __name__ == "__main__":
    # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)
    
    # Define dataset path
    dataset_path = "../data/DevanagariHandwrittenCharacterDataset/SmallDataSet"
    
    # Scenario 1: Train a new model from scratch
    # print("=== Training Example with Improved Data Loading ===")
    
    # Create model instance
    # model = CustomCNNModel(class_names=get_class_names(), input_shape=(32, 32, 3))
    
    # Train the model with improved data loading
    # history = model.train_with_generators(
    #     dataset_path=dataset_path,
    #     epochs=10,
    #     batch_size=32,
    #     learning_rate=0.0001,
    #     validation_split=0.2,
    #     save_best_weights=True,
    #     weights_save_path="model_checkpoints/improved_best_weights.weights.h5",
    #     early_stopping_patience=10,
    #     lr_reduce_patience=5,
    #     seed=42
    # )
    
    # Plot training history
    # model.plot_training_history("improved_training_plots.png")
    
    # Save the final model
    # model.save_model("saved_models/improved_custom_cnn_model.h5")
    
    # print("\n" + "="*50)
    
    # Scenario 2: Test with improved evaluation
    # print("=== Testing with Improved Evaluation ===")
    
    # # Test the model
    # test_results = model.evaluate_on_test_data(dataset_path, batch_size=32)
    
    # # Evaluate metrics
    # metrics = model.evaluate_metrics(
    #     test_results['predictions'], 
    #     test_results['true_labels'],
    #     class_names=model.class_names
    # )
    # print(f"\nFinal Test Accuracy: {test_results['accuracy']:.4f}")
    
    # print("\n" + "="*50)
    
    # # Scenario 3: Load pre-trained model and test
    # print("=== Loading Pre-trained Weights and Testing ===")
    weight_path = "model_checkpoints/improved_best_weights.weights.h5"
    
    # Load model with pre-trained weights
    model_pretrained = CustomCNNModel(
            weight_path=weight_path, 
            class_names=get_class_names(),
            input_shape=(32, 32, 3)
        )
        
    # Test the pre-trained model
    test_results_pretrained = model_pretrained.evaluate_on_test_data(dataset_path, batch_size=32)

    # test_results = model.evaluate_on_test_data(test_dir, batch_size=32)

    # Print overall accuracy
    # print(f"Overall accuracy on test set: {test_results['accuracy']*100:.2f}%")

    # Load predictions for detailed analysis
    results_df = pd.read_csv('test_predictions_improved.csv')

    # Extract true and predicted labels
    y_true = results_df['True Label']
    y_pred = results_df['Predicted Label']

    # Get unique class labels
    classes = sorted(y_true.unique())

    # Calculate overall metrics
    overall_accuracy = accuracy_score(y_true, y_pred)
    overall_precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nOverall Model Performance:")
    print(f"Accuracy: {overall_accuracy*100:.2f}%")
    print(f"Precision: {overall_precision:.4f}")
    print(f"Recall: {overall_recall:.4f}")
    print(f"F1-Score: {overall_f1:.4f}")

    # Calculate class-wise metrics
    class_wise_metrics = []

    for class_label in classes:
        # Create binary classification for current class vs all others
        y_true_binary = (y_true == class_label).astype(int)
        y_pred_binary = (y_pred == class_label).astype(int)
        
        # Calculate metrics for this class
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        
        # Count support (number of true instances)
        support = sum(y_true == class_label)
        
        class_wise_metrics.append({
            'Class': class_label,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Support': support
        })

    # Create DataFrame for class-wise metrics
    class_metrics_df = pd.DataFrame(class_wise_metrics)

    print("\nClass-wise Performance:")
    print(class_metrics_df.round(4))

    # Save class-wise metrics to CSV
    class_metrics_df.to_csv('class_wise_metrics.csv', index=False)
    print(f"\nClass-wise metrics saved to 'class_wise_metrics.csv'")

    # Create and save overall metrics dictionary
    overall_metrics = {
        'overall_accuracy': float(overall_accuracy),
        'overall_precision': float(overall_precision),
        'overall_recall': float(overall_recall),
        'overall_f1_score': float(overall_f1),
        'total_samples': len(y_true),
        'num_classes': len(classes),
        'class_names': classes
    }

    # Save overall metrics to JSON
    with open('overall_model_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)

    print(f"Overall model metrics saved to 'overall_model_metrics.json'")

    # Generate and save detailed classification report
    class_report = classification_report(y_true, y_pred, target_names=[str(cls) for cls in classes], output_dict=True)
    class_report_df = pd.DataFrame(class_report).transpose()
    class_report_df.to_csv('detailed_classification_report.csv')

    print(f"Detailed classification report saved to 'detailed_classification_report.csv'")

    # Generate and save confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)
    conf_matrix_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)
    conf_matrix_df.to_csv('confusion_matrix.csv')

    print(f"Confusion matrix saved to 'confusion_matrix.csv'")

    # Print summary of saved files
    print(f"\nSummary of saved files:")
    print(f"1. class_wise_metrics.csv - Individual class performance metrics")
    print(f"2. overall_model_metrics.json - Overall model performance summary")
    print(f"3. detailed_classification_report.csv - Sklearn classification report")
    print(f"4. confusion_matrix.csv - Confusion matrix")

    # Display top and bottom performing classes
    print(f"\nTop 3 performing classes (by F1-score):")
    top_classes = class_metrics_df.nlargest(3, 'F1-Score')[['Class', 'F1-Score', 'Support']]
    print(top_classes)

    print(f"\nBottom 3 performing classes (by F1-score):")
    bottom_classes = class_metrics_df.nsmallest(3, 'F1-Score')[['Class', 'F1-Score', 'Support']]
    print(bottom_classes)