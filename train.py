import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

def build_model():
    # Build a Convolutional Neural Network (CNN)
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(2, 2),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        
        Dense(1, activation='sigmoid')  # Binary classification: 1 output neuron with sigmoid activation
    ])
    
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(data_dir="dataset", model_save_path="face_mask_model.h5", epochs=10):
    # Set up image data generators for training and validation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    validation_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    # Generate data for training and validation
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',  # Binary classification (with mask or without mask)
        subset='training'
    )

    validation_generator = validation_datagen.flow_from_directory(
        data_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        subset='validation'
    )

    # Create and train the model
    model = build_model()
    history = model.fit(
        train_generator,
        validation_data=validation_generator,
        epochs=epochs
    )

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved at {model_save_path}")

if __name__ == "__main__":
    train_model()  # Start the training process
