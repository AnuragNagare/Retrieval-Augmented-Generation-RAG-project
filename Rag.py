

# import streamlit as st
# import numpy as np
# import tensorflow as tf
# import requests
# from PIL import Image
# import lime
# from lime import lime_image
# import matplotlib.pyplot as plt
# from skimage.segmentation import mark_boundaries

# # Set up paths to your CNN model
# cnn_model_path = 'D:/Dataset/skin_disease_cnn_model.h5'
# pubmed_api_key = 'e067a9b9ee2960205eb71e43428774056607'  # Replace with your PubMed API key

# # Load the CNN model
# @st.cache_resource
# def load_cnn_model():
#     return tf.keras.models.load_model(cnn_model_path)

# # Function to preprocess the image
# def preprocess_image(image):
#     image = image.resize((64, 64))  # Resize the image to the expected input size
#     img_array = np.array(image) / 255.0  # Normalize the image to range [0, 1]
#     img_array = np.expand_dims(img_array, axis=0).astype('float32')  # Add batch dimension and convert to float32
#     return img_array

# # Function to perform disease prediction using the CNN model
# def predict_disease(image, model):
#     img_array = preprocess_image(image)
#     prediction = model.predict(img_array)
#     disease_class = np.argmax(prediction)
#     return disease_class, prediction[0][disease_class]

# # Function to explain prediction using LIME
# def explain_prediction(image, model):
#     img_array = preprocess_image(image)[0]  # Use the preprocessed image without batch dimension for LIME
#     explainer = lime_image.LimeImageExplainer()
#     explanation = explainer.explain_instance(img_array, model.predict, top_labels=1, hide_color=0, num_samples=1000)
#     temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, hide_rest=False, min_weight=0.1)
#     return mark_boundaries(temp / 255.0, mask)

# # Function to search PubMed for articles related to the predicted disease
# def search_pubmed(disease_name):
#     base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
#     params = {
#         'db': 'pubmed',
#         'term': disease_name,
#         'retmax': 5,
#         'retmode': 'json',
#         'api_key': pubmed_api_key
#     }
#     response = requests.get(base_url, params=params)
#     article_ids = response.json().get('esearchresult', {}).get('idlist', [])
#     return fetch_pubmed_articles(article_ids) if article_ids else "No relevant articles found on PubMed."

# # Function to fetch article details from PubMed using ESummary
# def fetch_pubmed_articles(article_ids):
#     base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
#     params = {
#         'db': 'pubmed',
#         'id': ','.join(article_ids),
#         'retmode': 'json',
#         'api_key': pubmed_api_key
#     }
#     response = requests.get(base_url, params=params)
#     articles = response.json().get('result', {})
#     article_info = []
#     for article_id in article_ids:
#         article = articles.get(article_id, {})
#         title = article.get('title', 'No title available')
#         source = article.get('source', 'No source available')
#         url = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
#         article_info.append(f"**Title**: {title}\n**Source**: {source}\n[Read more]({url})\n")
#     return article_info

# # Streamlit app interface
# st.title('Skin Disease Detection and Information Retrieval')

# # Image upload section
# uploaded_image = st.file_uploader("Upload an image of the skin lesion", type=["jpg", "jpeg", "png"])

# if uploaded_image is not None:
#     image = Image.open(uploaded_image)
#     st.image(image, caption='Uploaded Image', use_column_width=True)

#     # Load the CNN model
#     cnn_model = load_cnn_model()

#     # Predict disease using the CNN model
#     disease_class, confidence = predict_disease(image, cnn_model)
#     st.write(f"Predicted Disease Class: {disease_class} with confidence: {confidence:.2f}")

#     # Explain prediction using LIME
#     st.write("LIME Explanation:")
#     explanation_image = explain_prediction(image, cnn_model)
#     st.image(explanation_image, caption='LIME Explanation', use_column_width=True)

#     # Retrieve PubMed articles related to the disease
#     disease_name = "Melanoma"  # Replace with your own logic to map disease_class to a disease name
#     st.write(f"**PubMed Articles Related to {disease_name}:**")
#     articles = search_pubmed(disease_name)
#     if articles:
#         for article in articles:
#             st.write(article)
#     else:
#         st.write("No relevant articles found on PubMed.")

import streamlit as st
import numpy as np
import tensorflow as tf
import requests
from PIL import Image

# Set up paths to your CNN model
cnn_model_path = 'D:/Dataset/skin_disease_cnn_model.h5'
pubmed_api_key = 'e067a9b9ee2960205eb71e43428774056607'  # Replace with your PubMed API key

# Define the labels (based on your dataset)
labels = ['Melanocytic nevi', 'Melanoma', 'Benign keratosis', 'Basal cell carcinoma',
          'Actinic keratoses', 'Vascular lesions', 'Dermatofibroma']

# Precautions, medications, and diet advice for each disease
# Expanded precautions, medications, and diet advice for each disease
disease_info = {
    'Melanocytic nevi': {
        'Precautions': "Avoid excessive sun exposure, wear broad-spectrum sunscreen with SPF 30 or higher, and seek shade whenever possible. Regularly examine the skin for any changes in size, shape, or color of the nevi, as these could be signs of potential malignancy. It's also advisable to avoid tanning beds and to wear protective clothing, such as hats and long sleeves, when outdoors.",
        'Medications': "No medications are typically needed unless the nevi show signs of change, such as rapid growth, irregular borders, or different colors. In such cases, a dermatologist may recommend surgical removal or a biopsy to rule out skin cancer. It's important to consult with a healthcare provider if you notice any unusual changes.",
        'Diet': "Maintain a healthy diet rich in antioxidants, including a variety of colorful fruits and vegetables like berries, oranges, spinach, and kale. Antioxidants can help combat skin damage caused by free radicals, which are often generated by UV exposure. Staying well-hydrated by drinking plenty of water is also beneficial for skin health."
    },
    'Melanoma': {
        'Precautions': "Perform regular self-examinations to check moles and other skin lesions for changes in size, color, or shape. Avoid prolonged sun exposure, especially between 10 a.m. and 4 p.m. when UV rays are strongest, and use a broad-spectrum sunscreen with an SPF of 50 or higher. Reapply sunscreen every two hours and after swimming or sweating. Wear sun-protective clothing, including a wide-brimmed hat and UV-blocking sunglasses.",
        'Medications': "Depending on the severity and stage of melanoma, treatment options may include topical creams for superficial lesions, immunotherapy drugs like pembrolizumab or nivolumab to boost the immune response, and surgical removal of the affected area. For advanced melanoma, targeted therapy drugs like BRAF or MEK inhibitors may be prescribed. Regular follow-ups with a dermatologist are essential.",
        'Diet': "Consume a diet rich in antioxidants to help repair skin damage. Include foods like berries, dark leafy greens, tomatoes, nuts, and seeds, which are known to reduce inflammation and support immune function. Omega-3 fatty acids found in fatty fish like salmon, chia seeds, and walnuts can also be beneficial. Limit intake of processed foods and sugar, as they can exacerbate inflammation."
    },
    'Benign keratosis': {
        'Precautions': "Avoid exposure to harsh chemicals and irritants that can aggravate the skin, and use gentle, fragrance-free skin care products. Moisturize daily with a cream or lotion containing ingredients like hyaluronic acid or ceramides to keep the skin barrier strong and hydrated. If you notice any itching or irritation, consult a dermatologist for appropriate care.",
        'Medications': "In most cases, no specific medications are needed. However, if the keratosis becomes bothersome or cosmetically concerning, topical treatments like retinoids or cryotherapy can be used to reduce the appearance of the lesion. Regularly using emollients can also help to soften the skin and reduce dryness.",
        'Diet': "Focus on a balanced diet that supports skin health, including foods rich in omega-3 fatty acids like flaxseeds, fish oil, and walnuts. Antioxidant-rich foods like citrus fruits, berries, and leafy greens can also help combat oxidative stress. Staying well-hydrated by drinking plenty of water throughout the day can promote skin elasticity and hydration."
    },
    'Basal cell carcinoma': {
        'Precautions': "Minimize sun exposure to prevent further damage. Always apply a broad-spectrum sunscreen with at least SPF 50, even on cloudy days. Avoid peak sun hours, wear wide-brimmed hats, long sleeves, and sunglasses that block UV rays. Regular skin checks are essential to catch any new or recurring lesions early.",
        'Medications': "Treatment may involve topical creams like imiquimod or fluorouracil for superficial lesions. For more invasive cases, radiation therapy or Mohs micrographic surgery may be recommended to remove the tumor while preserving as much healthy tissue as possible. Oral medications like vismodegib may be prescribed in advanced cases.",
        'Diet': "Incorporate foods high in antioxidants and anti-inflammatory properties, such as citrus fruits, green tea, dark chocolate, and vegetables like broccoli and kale. These foods may help support skin repair and reduce the risk of further skin damage. Avoid excessive alcohol and processed foods that can trigger inflammation."
    },
    'Actinic keratoses': {
        'Precautions': "Consistently use a broad-spectrum sunscreen with an SPF of 50 or more. Avoid direct sunlight during peak hours, and wear protective clothing, including long sleeves and wide-brimmed hats. Regularly examine your skin for any new patches or changes in existing lesions, and consult a dermatologist promptly if you notice anything unusual.",
        'Medications': "Common treatments include topical creams like 5-fluorouracil (Efudex) or imiquimod, which work by destroying abnormal cells. Cryotherapy, which involves freezing the lesion with liquid nitrogen, is another effective method. In some cases, laser therapy or photodynamic therapy may be used for more extensive areas.",
        'Diet': "Eat a diet that supports skin health, rich in vitamins A, C, and E, which are known for their antioxidant properties. Foods such as sweet potatoes, carrots, spinach, oranges, and almonds can help repair damaged skin cells and improve overall skin resilience. Staying hydrated is crucial for maintaining healthy skin texture and elasticity."
    },
    'Vascular lesions': {
        'Precautions': "Carefully monitor the lesions for any changes in size, color, or shape, and avoid trauma or injury to the affected areas. If the lesion is located on an area that is frequently rubbed or bumped, consider covering it with a soft bandage to protect it from irritation. Consult a healthcare provider if you notice any bleeding or growth in the lesion.",
        'Medications': "Most vascular lesions do not require medication unless they cause discomfort or bleeding. In such cases, laser therapy can be an effective option to reduce their appearance. In more severe cases, surgical interventions or sclerotherapy might be considered to manage symptoms.",
        'Diet': "Include anti-inflammatory foods in your diet, such as turmeric, ginger, fatty fish like salmon, and nuts like almonds and walnuts. These foods can help reduce inflammation in the body and may assist in the healing process. Limiting caffeine and alcohol can also be beneficial as they can exacerbate vascular issues."
    },
    'Dermatofibroma': {
        'Precautions': "Avoid scratching or picking at the lesion to prevent irritation or infection. If the dermatofibroma is in an area prone to friction or impact, consider using a protective pad or bandage to minimize discomfort. Regularly monitor for any changes in size, shape, or color, and consult a healthcare provider if any significant alterations occur.",
        'Medications': "Typically, no medications are required as dermatofibromas are benign and asymptomatic. If the lesion becomes painful or grows rapidly, surgical removal or steroid injections may be recommended to reduce its size. Over-the-counter pain relievers can help manage any associated discomfort.",
        'Diet': "Maintain a nutritious diet with a focus on high-fiber foods, lean proteins, and a variety of fruits and vegetables. Foods rich in vitamins C and E, such as citrus fruits, nuts, and green leafy vegetables, are particularly beneficial for skin healing and overall health. Staying hydrated by drinking plenty of water will also help keep your skin healthy and supple."
    }
}


# Load the CNN model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model(cnn_model_path)

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((64, 64))  # Resize the image to the expected input size
    img_array = np.array(image) / 255.0  # Normalize the image to range [0, 1]
    img_array = np.expand_dims(img_array, axis=0).astype('float32')  # Add batch dimension and convert to float32
    return img_array

# Function to perform disease prediction using the CNN model
def predict_disease(image, model):
    img_array = preprocess_image(image)
    prediction = model.predict(img_array)
    disease_index = np.argmax(prediction)
    disease_label = labels[disease_index]
    confidence = prediction[0][disease_index]
    return disease_label, confidence

# Function to search PubMed for articles related to the predicted disease
def search_pubmed(disease_name):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        'db': 'pubmed',
        'term': disease_name,
        'retmax': 5,
        'retmode': 'json',
        'api_key': pubmed_api_key
    }
    response = requests.get(base_url, params=params)
    article_ids = response.json().get('esearchresult', {}).get('idlist', [])
    return fetch_pubmed_articles(article_ids) if article_ids else "No relevant articles found on PubMed."

# Function to fetch article details from PubMed using ESummary
def fetch_pubmed_articles(article_ids):
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        'db': 'pubmed',
        'id': ','.join(article_ids),
        'retmode': 'json',
        'api_key': pubmed_api_key
    }
    response = requests.get(base_url, params=params)
    articles = response.json().get('result', {})
    article_info = []
    for article_id in article_ids:
        article = articles.get(article_id, {})
        title = article.get('title', 'No title available')
        source = article.get('source', 'No source available')
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
        article_info.append(f"**Title**: {title}\n**Source**: {source}\n[Read more]({url})\n")
    return article_info

# Streamlit app interface
st.title('Skin Disease Detection and Information Retrieval')

# Image upload section
uploaded_image = st.file_uploader("Upload an image of the skin lesion", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Load the CNN model
    cnn_model = load_cnn_model()

    # Predict disease using the CNN model
    disease_label, confidence = predict_disease(image, cnn_model)
    st.write(f"Predicted Disease: **{disease_label}** with confidence: {confidence:.2f}")

    # Retrieve PubMed articles related to the disease
    st.write(f"**PubMed Articles Related to {disease_label}:**")
    articles = search_pubmed(disease_label)
    if articles:
        for article in articles:
            st.write(article)
    else:
        st.write("No relevant articles found on PubMed.")

    # Display disease information (Precautions, Medications, Diet)
    if disease_label in disease_info:
        st.write(f"**Precautions for {disease_label}:** {disease_info[disease_label]['Precautions']}")
        st.write(f"**Medications for {disease_label}:** {disease_info[disease_label]['Medications']}")
        st.write(f"**Dietary Advice for {disease_label}:** {disease_info[disease_label]['Diet']}")
