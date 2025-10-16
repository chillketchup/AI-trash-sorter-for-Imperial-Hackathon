import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import argparse
import cv2

class TrashClassifier:
    def __init__(self, data_dir="trash-dataset", img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.model = None
        self.class_names = []
        self.history = None
        
    def load_and_prepare_data(self):
        """Load and prepare the dataset"""
        # Check if dataset exists
        if not os.path.exists(self.data_dir):
            raise FileNotFoundError(f"Dataset directory '{self.data_dir}' not found!")
        
        # Get class names from subdirectories
        self.class_names = sorted([d for d in os.listdir(self.data_dir)
                                 if os.path.isdir(os.path.join(self.data_dir, d))])
        
        if not self.class_names:
            raise ValueError(f"No subdirectories found in '{self.data_dir}'")
        
        print(f"Found {len(self.class_names)} classes: {self.class_names}")
        
        # Count images per class
        for class_name in self.class_names:
            class_path = os.path.join(self.data_dir, class_name)
            num_images = len([f for f in os.listdir(class_path)
                            if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"  {class_name}: {num_images} images")
        
        return self.class_names
    
    def create_data_generators(self, validation_split=0.2, batch_size=32):
        """Create data generators for training and validation"""
        datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        train_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        validation_generator = datagen.flow_from_directory(
            self.data_dir,
            target_size=self.img_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        return train_generator, validation_generator
    
    def build_model(self, num_classes):
        """Build the CNN model"""
        # Use a pre-trained model as base (transfer learning)
        base_model = tf.keras.applications.MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=(*self.img_size, 3)
        )
        
        # Freeze the base model
        base_model.trainable = False
        
        # Add custom layers
        model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train(self, epochs=20, batch_size=32, validation_split=0.2):
        """Train the model"""
        print("Loading data...")
        self.load_and_prepare_data()
        
        print("Creating data generators...")
        train_generator, validation_generator = self.create_data_generators(
            validation_split=validation_split,
            batch_size=batch_size
        )
        
        print("Building model...")
        self.build_model(len(self.class_names))
        
        print("Model summary:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        print("Starting training...")
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def save_model(self, model_path="trash_classifier_model.h5"):
        """Save the trained model"""
        if self.model:
            self.model.save(model_path)
            print(f"Model saved to {model_path}")
            
            # Save class names
            np.save("class_names.npy", self.class_names)
            print("Class names saved to class_names.npy")
        else:
            print("No model to save!")
    
    def load_model(self, model_path="trash_classifier_model.h5"):
        """Load a trained model"""
        if os.path.exists(model_path) and os.path.exists("class_names.npy"):
            self.model = keras.models.load_model(model_path)
            self.class_names = np.load("class_names.npy", allow_pickle=True)
            print(f"Model loaded from {model_path}")
            print(f"Classes: {self.class_names}")
        else:
            print("Model files not found!")
    
    def predict_image(self, image_path):
        """Predict a single image"""
        if self.model is None:
            self.load_model()
        
        # Load and preprocess image
        img = keras.preprocessing.image.load_img(image_path, target_size=self.img_size)
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        
        # Make prediction
        predictions = self.model.predict(img_array)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        
        return predicted_class, confidence, predictions[0]
    
    def predict_folder(self, folder_path):
        """Predict all images in a folder"""
        if self.model is None:
            self.load_model()
        
        results = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        
        for filename in os.listdir(folder_path):
            if filename.lower().endswith(valid_extensions):
                image_path = os.path.join(folder_path, filename)
                try:
                    predicted_class, confidence, _ = self.predict_image(image_path)
                    results.append({
                        'filename': filename,
                        'predicted_class': predicted_class,
                        'confidence': confidence
                    })
                    print(f"{filename}: {predicted_class} ({confidence:.2%})")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        return results
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main():
    parser = argparse.ArgumentParser(description='Trash Classification Model')
    parser.add_argument('--mode', choices=['train', 'predict', 'predict_folder'],
                       default='train', help='Mode to run')
    parser.add_argument('--image', type=str, help='Image path for prediction')
    parser.add_argument('--folder', type=str, help='Folder path for batch prediction')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    
    args = parser.parse_args()
    
    classifier = TrashClassifier()
    
    if args.mode == 'train':
        print("Training mode selected...")
        classifier.train(epochs=args.epochs, batch_size=args.batch_size)
        classifier.save_model()
        classifier.plot_training_history()
        
    elif args.mode == 'predict' and args.image:
        print("Single prediction mode...")
        classifier.load_model()
        if os.path.exists(args.image):
            predicted_class, confidence, _ = classifier.predict_image(args.image)
            print(f"\nPrediction: {predicted_class}")
            print(f"Confidence: {confidence:.2%}")
        else:
            print(f"Image not found: {args.image}")
            
    elif args.mode == 'predict_folder' and args.folder:
        print("Batch prediction mode...")
        classifier.load_model()
        if os.path.exists(args.folder):
            results = classifier.predict_folder(args.folder)
            print(f"\nProcessed {len(results)} images")
        else:
            print(f"Folder not found: {args.folder}")
    else:
        print("Please specify a valid mode and required arguments")

if __name__ == "__main__":
    main()
