import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Path to dataset and trained model
test_data_path = "BRAIN TUMOR DATASET/Testing"
model_path = "brhanced.h5"
output_csv = "brain_tumor___results.csv"

# Load the pre-trained model
model = load_model(model_path)

# Image preprocessing
test_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_generator = test_datagen.flow_from_directory(
    test_data_path,
    target_size=(256, 256),  # Update target size to match the model's input shape
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Perform prediction
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Calculate accuracy
accuracy = np.mean(predicted_classes == true_classes) * 100

# Calculate percentage of tumor detected
tumor_classes = [0, 1, 3]  # Assuming 'glioma', 'meningioma', and 'pituitary' represent tumors
tumor_count = sum(predicted_classes[i] in tumor_classes for i in range(len(predicted_classes)))
percentage_tumor_detected = (tumor_count / len(predicted_classes)) * 100

# Save results to CSV
results = pd.DataFrame({
    'Filename': test_generator.filenames,
    'True_Label': [class_labels[i] for i in true_classes],
    'Predicted_Label': [class_labels[i] for i in predicted_classes],
    'Prediction_Confidence': np.max(predictions, axis=1)
})
results.to_csv(output_csv, index=False)

# Print results
print(f"Accuracy: {accuracy:.2f}%")
print(f"Percentage of tumor detected: {percentage_tumor_detected:.2f}%")
print(f"Results saved to {output_csv}")
