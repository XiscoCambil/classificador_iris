from flask import Blueprint, jsonify, request
from services.knn_predict_service import knn_predict
import numpy as np

knn_controller = Blueprint('knn_controller', __name__)

@knn_controller.route('/knn_predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    new_flower = np.array(data['new_flower']).reshape(1, -1)

    prediction = knn_predict(new_flower)

    res = "Iris Setosa (target 0)" if prediction[0] == 0 else ( "Iris Versicolour (target 1)" if prediction[0] == 1 else "Iris Virginica (target 2)")

    return jsonify({'prediccion': res})