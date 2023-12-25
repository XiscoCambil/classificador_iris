from flask import Blueprint, jsonify, request
from services.decision_tree_predict_service import decision_tree_predict
import numpy as np

decision_tree_controller = Blueprint('decision_tree_controller', __name__)

@decision_tree_controller.route('/decision_tree_predict', methods=['POST'])
def predict():
    
    data = request.get_json()

    new_flower = np.array(data['new_flower']).reshape(1, -1)

    prediction = decision_tree_predict(new_flower)
   
    return jsonify({'prediccion': int(prediction[0])})