import cv2
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score
import pandas as pd

# Load pre-trained model (replace with your model path)
model = load_model('brain_tumor_cnn_model_enhanced.h5')

# Tumor information dictionary
tumor_info = {
    'glioma': {
        'description': '''Glioma is a type of tumor that originates in the glial cells of the brain or spinal cord. Glial cells are non-neuronal cells that support and protect neurons. Gliomas are classified based on their aggressiveness and location. The most common forms of glioma include astrocytomas, oligodendrogliomas, and ependymomas. Symptoms of gliomas vary depending on the tumor’s size, location, and rate of growth. Common symptoms may include headaches, nausea, seizures, vision problems, and cognitive changes. Gliomas can be benign or malignant, with malignant gliomas being highly aggressive and invasive, making them difficult to treat. Early detection through imaging techniques such as MRI is crucial for determining the best course of action.''',
        'treatment': '''The treatment for gliomas often includes a combination of surgery, radiation therapy, and chemotherapy. Surgery aims to remove as much of the tumor as possible without damaging surrounding brain tissue. Radiation therapy is commonly used after surgery to target remaining tumor cells. Chemotherapy may also be administered to shrink the tumor or prevent regrowth. In some cases, targeted therapy and immunotherapy may be used to attack cancer cells more precisely. Treatment plans depend on factors like tumor grade, location, and the patient’s overall health.''',
        'precautions': '''People diagnosed with gliomas should undergo regular check-ups to monitor the progress of treatment and detect any recurrence of the tumor. It is also important to avoid exposure to chemicals and other environmental toxins that may contribute to cancer development. Maintaining a healthy lifestyle, including regular physical activity, a balanced diet, and stress management, can support overall health and improve quality of life during treatment. Early diagnosis through advanced imaging technologies plays a vital role in managing glioma effectively.''',
    },
    'meningioma': {
        'description': '''Meningiomas are tumors that develop in the meninges, the protective layers of tissue surrounding the brain and spinal cord. These tumors are typically benign, but they can also be malignant in rare cases. Meningiomas grow slowly and may remain asymptomatic for years, making them difficult to detect until they reach a certain size. As they grow, they may press against nearby brain tissue, nerves, and blood vessels, causing symptoms such as headaches, seizures, vision changes, hearing loss, and memory issues. Meningiomas are often found incidentally during scans for other conditions. The majority of meningiomas are treated successfully with surgery, though in some cases, radiation therapy or medications may be necessary.''',
        'treatment': '''The primary treatment for meningiomas is surgical removal of the tumor. If the tumor is in a location that is difficult to access or if it has grown too large, radiation therapy may be used to shrink the tumor before or after surgery. In cases where surgery is not feasible, radiation therapy is the main treatment to stop tumor growth. Medications, such as steroids, may be prescribed to reduce inflammation or control symptoms like seizures. If the meningioma is not causing symptoms and is small, doctors may choose a watch-and-wait approach, monitoring the tumor with periodic imaging.''',
        'precautions': '''Early diagnosis through imaging, such as MRI or CT scans, is crucial in managing meningiomas. Regular follow-up exams are necessary for patients who have undergone treatment to ensure the tumor does not recur. Stress management, healthy nutrition, and lifestyle changes, such as quitting smoking, are also recommended to reduce the risk of developing further health issues. Patients with meningiomas are often advised to lead a healthy lifestyle, with an emphasis on mental well-being and regular physical activity, to help manage the emotional and physical impacts of the tumor and treatment.''',
    },
    'pituitary': {
        'description': '''Pituitary tumors are abnormal growths that form in the pituitary gland, which is located at the base of the brain and regulates many important hormones that control vital body functions. These tumors can be benign (non-cancerous) or malignant (cancerous), with benign tumors being far more common. Symptoms of pituitary tumors are often related to hormonal imbalances caused by the tumor's effect on hormone production. Symptoms may include unexplained weight gain or loss, changes in menstrual cycles, fatigue, headaches, vision problems, and growth abnormalities. Pituitary tumors can also cause hyperprolactinemia, acromegaly, or Cushing’s disease, depending on the type of tumor and hormones involved.''',
        'treatment': '''The treatment for pituitary tumors typically involves surgery to remove the tumor. In some cases, surgery is performed through the nose (transsphenoidal surgery) to minimize damage to surrounding structures. Radiation therapy is sometimes used when surgery is not possible, or if the tumor recurs. Hormone therapy is used to treat imbalances caused by the tumor. Medications that regulate hormone levels, such as dopamine agonists or somatostatin analogs, can be prescribed to shrink tumors and alleviate symptoms. In rare cases, if the tumor is malignant, chemotherapy may be considered.''',
        'precautions': '''People with pituitary tumors should be closely monitored to ensure their hormonal levels remain balanced and that the tumor does not grow back. Regular eye exams are also important, as pituitary tumors can affect vision due to their proximity to the optic nerves. Maintaining a healthy diet and avoiding excessive stress are also essential in managing the condition. In addition, patients should have regular check-ups to ensure any hormonal imbalances are detected early and treated effectively.''',
    },
    'no tumor': {
        'description': '''No tumor detected means that the brain scans (such as MRI or CT) show no signs of abnormal growth or mass. This is the most favorable result, as it indicates the absence of any brain tumors. It is important to note that, while no tumor has been detected, other neurological or medical conditions may still exist, so symptoms like headaches or dizziness should not be ignored. In the absence of a tumor, it’s important to continue with regular health screenings to detect other potential conditions early.''',
        'treatment': '''Since no tumor is present, no medical treatment is necessary. However, for overall brain health and well-being, it is recommended to maintain a healthy lifestyle. A balanced diet rich in nutrients that support brain function, regular physical activity, and adequate rest are essential for maintaining optimal brain health. It is also important to manage stress effectively and avoid behaviors that could negatively affect cognitive and mental health, such as smoking or excessive alcohol consumption.''',
        'precautions': '''Maintaining a healthy lifestyle is key to preserving brain health. Regular exercise, a balanced diet rich in omega-3 fatty acids, antioxidants, and other brain-boosting nutrients, along with adequate sleep, can contribute to cognitive longevity. Additionally, keeping stress levels in check through mindfulness or relaxation techniques is beneficial. Regular check-ups with a healthcare provider, even without symptoms, ensure that any future health concerns are detected early. It's also advisable to avoid head injuries and follow safety protocols in physical activities to protect the brain.''',
    }
}


# Function to predict tumor type and provide details
def detect_tumor(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    class_idx = np.argmax(prediction, axis=1)[0]

    class_labels = ['glioma', 'meningioma', 'pituitary', 'no tumor']
    tumor_label = class_labels[class_idx]

    details = tumor_info.get(tumor_label, {})

    return tumor_label, details


image_path = 'BRAIN TUMOR DATASET/Testing/meningioma/Te-me_0022.jpg'
tumor_label, details = detect_tumor(image_path)

print(f'Tumor Type: {tumor_label}')
print('Description:', details['description'])
print('Treatment:', details['treatment'])
print('Precautions:', details['precautions'])

results = {
    'image': [image_path],
    'tumor_type': [tumor_label],
    'description': [details['description']],
    'treatment': [details['treatment']],
    'precautions': [details['precautions']]
}
df = pd.DataFrame(results)
df.to_csv('tumor_detection_results.csv', index=False)
