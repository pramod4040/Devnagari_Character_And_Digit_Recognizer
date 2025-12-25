import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB3
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
from PIL import Image
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd


class EnhancedMoeModel:
    def __init__(self, weight_path=None, class_names=None):
        self.weight_path = weight_path
        self.model = None
        self.class_names = class_names
        self.input_shape = (32, 32, 3)
        self.num_classes = 46 if class_names is None else len(class_names)
        self.num_experts = 3
        
        # Create the model architecture
        self.create_model(
            input_shape=self.input_shape, 
            num_classes=self.num_classes, 
            num_experts=self.num_experts
        )
        
        # Load weights if path is provided
        if self.weight_path and os.path.exists(self.weight_path):
            self.load_weights()
    
    def create_efficientnet_b0_expert(self, input_shape, num_classes, name_prefix="expert_b0"):
        input_tensor = Input(shape=input_shape)
        resized_input = layers.Resizing(224, 224)(input_tensor)
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_tensor=resized_input
        )
        
        # Fine-tune the last few layers
        for layer in base_model.layers[:-20]:
            layer.trainable = False
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.3)(x)
        output_tensor = Dense(num_classes, activation='softmax', name=f'{name_prefix}_output')(x)
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')

    def create_efficientnet_b3_expert(self, input_shape, num_classes, name_prefix="expert_b3"):
        input_tensor = Input(shape=input_shape)
        resized_input = layers.Resizing(300, 300)(input_tensor)
        base_model = EfficientNetB3(
            include_top=False,
            weights='imagenet',
            input_tensor=resized_input
        )
        
        # Fine-tune the last few layers
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        
        x = GlobalAveragePooling2D()(base_model.output)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = Dropout(0.4)(x)
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.3)(x)
        output_tensor = Dense(num_classes, activation='softmax', name=f'{name_prefix}_output')(x)
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')

    def create_custom_cnn_expert(self, input_shape, num_classes, name_prefix="expert_cnn"):
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
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name=f'{name_prefix}')

    def create_enhanced_gating_network(self, input_shape, num_experts):
        input_tensor = Input(shape=input_shape)
        
        # Convolutional layers for feature extraction
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(input_tensor)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        output_tensor = layers.Dense(num_experts, activation='softmax')(x)
        
        return models.Model(inputs=input_tensor, outputs=output_tensor, name="enhanced_gating")

    def create_model(self, input_shape, num_classes, num_experts):
        input_layer = Input(shape=input_shape)
        
        # Create diverse experts
        expert1 = self.create_efficientnet_b0_expert(input_shape, num_classes, name_prefix="expert1")
        expert2 = self.create_efficientnet_b3_expert(input_shape, num_classes, name_prefix="expert2")
        expert3 = self.create_custom_cnn_expert(input_shape, num_classes, name_prefix="expert3")
        experts = [expert1, expert2, expert3]
        
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

    # def load_weights(self):
    #     """Load model weights from the specified path"""
    #     try:
    #         # Load weights by name, ignoring missing layers or shape mismatch
    #         self.model.load_weights(self.weight_path, by_name=True, skip_mismatch=True)
    #         print(f"Successfully loaded weights from {self.weight_path}")
            
    #         # Compile the model after loading weights
    #         optimizer = Adam(learning_rate=0.001)
    #         self.model.compile(
    #             optimizer=optimizer,
    #             loss='categorical_crossentropy',
    #             metrics=['accuracy']
    #         )
                        
    #     except Exception as e:
    #         print(f"Error loading weights: {e}")

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
        optimizer = Adam(learning_rate=0.001)
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
        'digit_9'
    ]
    return class_names

# Example of how to use the model
if __name__ == "__main__":
    # Path to your saved model weights
    weights_path = os.path.join(
                os.getcwd(), 
                "../trainModel", 
                "model_checkpoints", 
                "moe_optimized_model-23-0.9473.weights.h5"
            )
    
    test_dir = os.path.join(os.getcwd(), "../data", "DevanagariHandwrittenCharacterDataset", "SmallDataSet")
    
    # Create model instance and load weights
    model = EnhancedMoeModel(weight_path=weights_path, class_names=get_class_names())
    
    # Optional: Compile the model with custom learning rate
    # Run prediction on all test images
    # test_results = model1.predictTest(test_dir, batch_size=3)
    test_results = model.evaluate_on_test_data(test_dir, batch_size=32)

    # Print overall accuracy
    print(f"Overall accuracy on test set: {test_results['accuracy']*100:.2f}%")

    # Load predictions for detailed analysis
    results_df = pd.read_csv('test_predictions_improved.csv')

    # Extract true and predicted labels
    y_true = results_df['True Label']
    y_pred = results_df['Predicted Label']

    # Get unique class labels
    classes = sorted(y_true.unique())

    # Calculate overall metrics - Enhanced with macro and micro averages
    overall_accuracy = accuracy_score(y_true, y_pred)
    
    # Weighted averages (original)
    overall_precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    overall_f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Macro averages (unweighted mean of per-class metrics)
    overall_precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    overall_recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    overall_f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Micro averages (aggregate the contributions across all classes)
    overall_precision_micro = precision_score(y_true, y_pred, average='micro', zero_division=0)
    overall_recall_micro = recall_score(y_true, y_pred, average='micro', zero_division=0)
    overall_f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)

    print(f"\nOverall Model Performance:")
    print(f"Accuracy: {overall_accuracy*100:.2f}%")
    print(f"\nWeighted Averages:")
    print(f"Precision (Weighted): {overall_precision_weighted:.4f}")
    print(f"Recall (Weighted): {overall_recall_weighted:.4f}")
    print(f"F1-Score (Weighted): {overall_f1_weighted:.4f}")
    print(f"\nMacro Averages:")
    print(f"Precision (Macro): {overall_precision_macro:.4f}")
    print(f"Recall (Macro): {overall_recall_macro:.4f}")
    print(f"F1-Score (Macro): {overall_f1_macro:.4f}")
    print(f"\nMicro Averages:")
    print(f"Precision (Micro): {overall_precision_micro:.4f}")
    print(f"Recall (Micro): {overall_recall_micro:.4f}")
    print(f"F1-Score (Micro): {overall_f1_micro:.4f}")

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

    # Create and save enhanced overall metrics dictionary
    overall_metrics = {
        'overall_accuracy': float(overall_accuracy),
        
        # Weighted averages
        'overall_precision_weighted': float(overall_precision_weighted),
        'overall_recall_weighted': float(overall_recall_weighted),
        'overall_f1_score_weighted': float(overall_f1_weighted),
        
        # Macro averages
        'overall_precision_macro': float(overall_precision_macro),
        'overall_recall_macro': float(overall_recall_macro),
        'overall_f1_score_macro': float(overall_f1_macro),
        
        # Micro averages
        'overall_precision_micro': float(overall_precision_micro),
        'overall_recall_micro': float(overall_recall_micro),
        'overall_f1_score_micro': float(overall_f1_micro),
        
        # Additional metadata
        'total_samples': len(y_true),
        'num_classes': len(classes),
        'class_names': classes
    }

    # Save overall metrics to JSON
    with open('overall_model_metrics.json', 'w') as f:
        json.dump(overall_metrics, f, indent=4)

    print(f"Enhanced overall model metrics saved to 'overall_model_metrics.json'")

    # Create a summary table for better visualization
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Weighted': [overall_accuracy, overall_precision_weighted, overall_recall_weighted, overall_f1_weighted],
        'Macro': [overall_accuracy, overall_precision_macro, overall_recall_macro, overall_f1_macro],
        'Micro': [overall_accuracy, overall_precision_micro, overall_recall_micro, overall_f1_micro]
    })
    
    print(f"\nMetrics Summary Table:")
    print(metrics_summary.round(4))
    
    # Save metrics summary table
    metrics_summary.to_csv('metrics_summary_table.csv', index=False)
    print(f"Metrics summary table saved to 'metrics_summary_table.csv'")

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
    print(f"2. overall_model_metrics.json - Enhanced overall model performance summary")
    print(f"3. metrics_summary_table.csv - Summary table of all averaging methods")
    print(f"4. detailed_classification_report.csv - Sklearn classification report")
    print(f"5. confusion_matrix.csv - Confusion matrix")

    # Display top and bottom performing classes
    print(f"\nTop 3 performing classes (by F1-score):")
    top_classes = class_metrics_df.nlargest(3, 'F1-Score')[['Class', 'F1-Score', 'Support']]
    print(top_classes)

    print(f"\nBottom 3 performing classes (by F1-score):")
    bottom_classes = class_metrics_df.nsmallest(3, 'F1-Score')[['Class', 'F1-Score', 'Support']]
    print(bottom_classes)