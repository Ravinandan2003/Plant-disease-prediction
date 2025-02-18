import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model_quantized.tflite")
interpreter.allocate_tensors()

# Set up input and output tensors
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Prediction function
def predict(input_data):
    input_data = input_data.astype('float32')
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Simple prediction route
@app.route('/predict', methods=['POST'])
def predict_route():
    input_data = np.array(request.json['data'])
    prediction = predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
