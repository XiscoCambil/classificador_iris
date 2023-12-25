from flask import Blueprint, jsonify, request
from services.knn_predict_service import knn_predict
import numpy as np

knn_controller = Blueprint('knn_controller', __name__)

@knn_controller.route('/knn_predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    new_flower = np.array(data['new_flower']).reshape(1, -1)

    prediction = knn_predict(new_flower)

    return jsonify({'prediccion': int(prediction[0])})