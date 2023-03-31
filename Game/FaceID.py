import streamlit as st
import cv2
import tensorflow as tf
import numpy as np


# Charger le modèle d'apprentissage en profondeur pré-entrainé pour la reconnaissance faciale
model = tf.keras.models.load_model("modele_reconnaissance_faciale.h5")

# Créer une zone de saisie de fichier pour télécharger l'image à reconnaître
uploaded_file = st.file_uploader("Télécharger une image de visage...", type=["jpg", "jpeg", "png"])

# Si une image est téléchargée
if uploaded_file is not None:
    # Charger l'image dans un tableau numpy et la convertir en échelle de gris
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Détecter les visages dans l'image
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)

    # Pour chaque visage détecté, faire une prédiction
    for (x, y, w, h) in faces:
        # Extraire la zone du visage
        roi = gray_image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = tf.keras.preprocessing.image.img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        # Faire une prédiction avec le modèle
        predictions = model.predict(roi)[0]
        emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
        max_index = np.argmax(predictions)
        emotion = emotion_labels[max_index]

        # Dessiner un rectangle autour du visage détecté et afficher l'émotion prédite
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Afficher l'image résultante
    st.image(image, caption='Image résultante', use_column_width=True)
