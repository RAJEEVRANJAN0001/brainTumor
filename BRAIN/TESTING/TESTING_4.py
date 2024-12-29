import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pyttsx3
from PIL import Image
from datetime import datetime

# Initialize Text-to-Speech Engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

# Add custom background and other styles for the main page and sidebar
st.markdown(
    """
    <style>
    body {
        background-color: transparent;  /* No background color for the main page */
    }
    .reportview-container {
        background-color: transparent; /* No background color for the report container */
        padding: 20px;
        color: #333;
    }
    .sidebar .sidebar-content {
        background: transparent; /* Sidebar with no background */
        color: #333;
    }
    h1, h2, h3 {
        color: #FFD700;  /* Gold color for headings */
    }
    .stButton>button {
        background-color: #4CAF50; /* Green button */
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #45a049; /* Darker green on hover */
    }
    .stTextInput>div>input {
        background-color: #f4f4f9;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        color: #333;
    }
    .stRadio>div {
        background-color: #f4f4f9;
        border: 2px solid #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }
    .stDownloadButton>button {
        background-color: #2196F3;  /* Blue download button */
        color: white;
        font-size: 16px;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stDownloadButton>button:hover {
        background-color: #0b7dda; /* Darker blue on hover */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Display the image on the main page
image_path = 'Unknown.jpeg'  # Path to the uploaded image
st.image(image_path, caption="Brain Tumor Detection", use_container_width=True)

# Add title and description
st.title("Brain Tumor Detection")
st.markdown("This web application detects brain tumors from MRI images and provides a detailed report with tumor classification.")

# Patient Information Form (Moved to the main page)
st.markdown("### Patient Information")

patient_no = st.text_input("Patient No:")
patient_name = st.text_input("Patient Name:")
dob = st.date_input("Date of Birth", min_value=datetime(1900, 1, 1))
patient_sex = st.radio("Sex:", ("Male", "Female", "Other"))
uploaded_file = st.file_uploader("Upload Brain MRI Image", type=["jpg", "png", "jpeg"])
clinical_history = st.file_uploader("Upload Clinical History (Optional)", type=["pdf", "docx", "txt"])

# Try loading the model and handle errors
def load_model_safe(model_path):
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

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

# Function to detect tumor
def detect_tumor(image_data, model):
    image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (256, 256))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0

    prediction = model.predict(image)
    class_idx = np.argmax(prediction, axis=1)[0]
    tumor_percentage = round(prediction[0][class_idx] * 100, 2)

    class_labels = ['glioma', 'meningioma', 'pituitary', 'no tumor']
    tumor_label = class_labels[class_idx]
    details = tumor_info.get(tumor_label, {})

    if tumor_percentage <= 25:
        stage = "1st Stage"
    elif tumor_percentage <= 50:
        stage = "2nd Stage"
    else:
        stage = "3rd Stage"

    return tumor_label, details, tumor_percentage, stage

# Function to mark tumor (mock implementation)
def mark_tumor(image_data):
    image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    center = (w // 2, h // 2)
    radius = min(h, w) // 4
    marked_image = cv2.circle(image.copy(), center, radius, (0, 0, 255), 4)
    return marked_image

# Function to create a report
def generate_report(patient_no, patient_name, dob, tumor_label, details, tumor_percentage, stage, clinical_history=None):
    report = f"""
    Brain Tumor Detection Report
    ----------------------------

    Patient Number: {patient_no}
    Patient Name: {patient_name}
    Date of Birth: {dob}
    
    Clinical History: {clinical_history if clinical_history else 'Not Provided'}
    
    Diagnosis:
    Tumor Type: {tumor_label}
    Stage: {stage} ({tumor_percentage}%)

    Description:
    {details['description']}

    Treatment:
    {details['treatment']}

    Precautions:
    {details['precautions']}
    """
    return report

# Load model
model_path = 'BRAIN TUMOR/DenseNet121+CNN/brain_tumor_cnn_model_enhanced.h5'
model = load_model_safe(model_path)

if st.button("Detect Tumor"):
    if uploaded_file is not None and model:
        image_data = Image.open(uploaded_file)
        tumor_label, details, tumor_percentage, stage = detect_tumor(image_data, model)

        st.subheader("Detection Results")
        st.image(image_data, caption="Uploaded Image", use_container_width=True)

        marked_image = mark_tumor(image_data)
        st.image(marked_image, caption="Tumor Marked Image", use_container_width=True)

        st.write(f"**Tumor Type:** {tumor_label}")
        st.write(f"**Stage:** {stage} ({tumor_percentage}%)")
        st.write("**Description:**")
        st.write(details['description'])
        st.write("**Treatment:**")
        st.write(details['treatment'])

        report = generate_report(patient_no, patient_name, dob, tumor_label, details, tumor_percentage, stage, clinical_history.name if clinical_history else None)
        st.subheader("Generated Report")
        st.text(report)

        report_path = "/tmp/brain_tumor_report.txt"
        with open(report_path, "w") as f:
            f.write(report)

        st.download_button(
            label="Download Report",
            data=open(report_path, "rb").read(),
            file_name="brain_tumor_report.txt",
            mime="text/plain"
        )

        engine.say(f'Tumor Type: {tumor_label}')
        engine.say(f'Stage: {stage} with {tumor_percentage}% involvement')
        engine.say('Description: ' + details['description'])
        engine.say('Treatment: ' + details['treatment'])
        engine.runAndWait()
    else:
        st.error("Please upload a brain MRI image and ensure the model is loaded correctly.")
