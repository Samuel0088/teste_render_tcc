from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os

app = Flask(__name__)

model = load_model("treinamento_para_servidor.h5")

classes = [
    "acaros_de_duas_manchas",
    "enrolamento_de_folha",
    "folhas_saudaveis",
    "mancha_alvo",
    "mancha_bacteriana",
    "mancha_septoria",
    "pinta_preta",
    "requeima",
    "virus_folha_amarela",
    "virus_mosaico_do_tomate_Y"
]

@app.route("/")
def home():
    return "API de Reconhecimento de Doenças em Folhas 🌿"

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    
    img = image.load_img(file, target_size=(160,160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]

    return jsonify({
        "classe": predicted_class
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
