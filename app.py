from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite

app = Flask(__name__)

# Carrega modelo TFLite
interpreter = tflite.Interpreter(model_path="modelo.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

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
    
    img = Image.open(file).resize((160,160))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = classes[np.argmax(output)]

    return jsonify({"classe": predicted_class})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
