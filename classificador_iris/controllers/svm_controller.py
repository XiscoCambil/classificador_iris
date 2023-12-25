from flask import Blueprint, jsonify, request
from services.svm_predicts_service import svm_predict
import numpy as np

svm_controller = Blueprint('svm_controller', __name__)

@svm_controller.route('/svm_predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    new_flower = np.array(data['new_flower']).reshape(1, -1)

    prediction = svm_predict(new_flower)
    
    return jsonify({'prediccion': int(prediction[0])})