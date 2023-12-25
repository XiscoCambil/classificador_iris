from flask import Flask
from controllers.regression_logistic_controller import regression_logistic_controller
from controllers.svm_controller import svm_controller
from controllers.decision_tree_controller import decision_tree_controller
from controllers.knn_controller import knn_controller

app = Flask('flower-predict')
app.register_blueprint(regression_logistic_controller)
app.register_blueprint(svm_controller)
app.register_blueprint(decision_tree_controller)
app.register_blueprint(knn_controller)

if __name__ == '__main__':
    app.run(debug=True, port=8000)