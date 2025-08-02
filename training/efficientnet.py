# 1. Import Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, image
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from google.colab import files

# 2. Define Paths and Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
CLASS_NAMES = ['rice', 'non_rice', 'non_leaf']
DISPLAY_NAMES = {'rice': 'RICE LEAF', 'non_rice': 'NOT A RICE LEAF', 'non_leaf': 'NOT A LEAF'}
DATASET_PATH = '/content/dataset'
CHECKPOINT_PATH = '/content/drive/MyDrive/rice_leaf_classifier_best_weights.keras'
FINAL_MODEL_PATH = '/content/drive/MyDrive/rice_leaf_classifier_model_final.keras'
CSV_LOG_PATH = '/content/drive/MyDrive/training_log.csv'

# 3. Data Generators
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    horizontal_flip=True,
    rotation_range=10,
    zoom_range=0.1
)

train_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False  # Important for evaluation
)

# 4. Model Creation and Compilation
base_model = EfficientNetB0(include_top=False, weights='imagenet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
base_model.trainable = True  # Enable fine-tuning

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
output = Dense(3, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5. Load Checkpoint Weights if available
if os.path.exists(CHECKPOINT_PATH):
    print("🔁 Loading previous best weights...")
    model.load_weights(CHECKPOINT_PATH)

# 6. Training Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy'),
    CSVLogger(CSV_LOG_PATH)
]

# 7. Train the Model
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

# 8. Save Final Model
model.save(FINAL_MODEL_PATH)
print(f"\n✅ Final model saved to: {FINAL_MODEL_PATH}")

# 9. Evaluate on Validation Set
loss, acc = model.evaluate(val_gen)
print("\n📊 Final Validation Accuracy: {:.2f}%".format(acc * 100))

# 10. Plot Training Curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.legend()

plt.show()

# 11. Classification Report & Confusion Matrix
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen, verbose=1)
y_pred = np.argmax(y_pred_probs, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=[DISPLAY_NAMES[c] for c in CLASS_NAMES]))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES],
            yticklabels=[DISPLAY_NAMES[c] for c in CLASS_NAMES])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("✅ Final Validation Accuracy: {:.2f}%".format(accuracy_score(y_true, y_pred) * 100))

# 12. Upload & Predict Custom Image
print("\n📂 Upload an image to classify...")
uploaded = files.upload()

model = load_model(FINAL_MODEL_PATH)

for fn in uploaded.keys():
    try:
        img = image.load_img(fn, target_size=(IMAGE_SIZE, IMAGE_SIZE))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)[0]
        class_index = np.argmax(prediction)
        label = CLASS_NAMES[class_index]
        confidence = prediction[class_index] * 100

        print(f"\n✅ Prediction for '{fn}': {DISPLAY_NAMES[label]} ({confidence:.2f}%)")
    except Exception as e:
        print(f"\n❌ Failed to process '{fn}': {e}")
