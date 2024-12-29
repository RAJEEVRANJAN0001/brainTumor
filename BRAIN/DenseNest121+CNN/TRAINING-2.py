import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
from sklearn.preprocessing import label_binarize
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

IMG_HEIGHT = 256  # Increased to avoid shrinking too much
IMG_WIDTH = 256
BATCH_SIZE = 32

train_dir = '/Training'
test_dir = '/Testing'

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

base_model_densenet = DenseNet121(weights=None, include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model_densenet.load_weights('/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5')  # Manually downloaded weights

cnn_model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(4, activation='softmax')
])

combined_input = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

densenet_output = base_model_densenet(combined_input)
densenet_output = layers.GlobalAveragePooling2D()(densenet_output)
densenet_output = layers.Dense(512, activation='relu')(densenet_output)

cnn_output = cnn_model(combined_input)

merged_output = layers.concatenate([densenet_output, cnn_output])

final_output = layers.Dense(4, activation='softmax')(merged_output)

model = models.Model(inputs=combined_input, outputs=final_output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('brain_tumor_cnn_best_model.keras', save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
]

steps_per_epoch = 60
validation_steps = 60

history = model.fit(
    train_generator,
    epochs=25,  # Set the desired number of epochs
    steps_per_epoch=steps_per_epoch,  # Set steps per epoch to 80
    validation_data=test_generator,
    validation_steps=validation_steps,  # Set validation steps to 80
    verbose=1,
    callbacks=callbacks
)

model.save('/brain_tumor_cnn_model_enhanced.h5')

model.save_weights('/brain_tumor_cnn_weights_enhanced.weights.h5')

test_loss, test_acc = model.evaluate(test_generator)
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

model.save('brain_tumor_cnn_saved_model_enhanced')

true_labels = test_generator.classes
predictions = model.predict(test_generator, verbose=1)

true_labels_one_hot = label_binarize(true_labels, classes=[0, 1, 2, 3])

fpr, tpr, _ = roc_curve(true_labels_one_hot.ravel(), predictions.ravel())
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.savefig('roc_curve.png')
plt.show()

cm = confusion_matrix(true_labels, np.argmax(predictions, axis=1))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plot as a file
plt.savefig('accuracy_loss_plot_enhanced.png')
plt.show()
