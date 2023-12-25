from flask import Blueprint, jsonify, request
from services.regression_logistic_predicts_service import regression_logistic_predict
import numpy as np

regression_logistic_controller = Blueprint('regression_logistic_controller', __name__)

@regression_logistic_controller.route('/regression_logistic_predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    new_flower = np.array(data['new_flower']).reshape(1, -1)

    prediction = regression_logistic_predict(new_flower)

    res = "Iris Setosa (target 0)" if prediction[0] == 0 else ( "Iris Versicolour (target 1)" if prediction[0] == 1 else "Iris Virginica (target 2)")
    
    return jsonify({'prediccion': res})