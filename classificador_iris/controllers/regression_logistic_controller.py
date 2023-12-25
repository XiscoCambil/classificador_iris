from flask import Blueprint, jsonify, request
from services.regression_logistic_predicts_service import regression_logistic_predict
import numpy as np

regression_logistic_controller = Blueprint('regression_logistic_controller', __name__)

@regression_logistic_controller.route('/regression_logistic_predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    new_flower = np.array(data['new_flower']).reshape(1, -1)

    prediction = regression_logistic_predict(new_flower)
    # Devolver la predicción como JSON
    return jsonify({'prediccion': int(prediction[0])})