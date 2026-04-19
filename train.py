import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from sklearn.utils.class_weight import compute_class_weight
import config

# -----------------------
# Data Generators
# -----------------------
train_datagen = ImageDataGenerator(
    preprocessing_function=tf.keras.applications.efficientnet.preprocess_input,
    rotation_range=25,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

train_gen = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='validation'
)

# -----------------------
# Class Weights
# -----------------------
labels = train_gen.classes
weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
class_weights = dict(enumerate(weights))

# -----------------------
# Model
# -----------------------
base_model = EfficientNetB3(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(config.NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------
# Train (Phase 1)
# -----------------------
model.fit(train_gen, validation_data=val_gen, epochs=10, class_weight=class_weights)

# -----------------------
# Fine-tuning
# -----------------------
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(train_gen, validation_data=val_gen, epochs=5, class_weight=class_weights)

# Save model
model.save(config.MODEL_PATH)