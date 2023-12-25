from flask import Flask
from controllers.regression_logistic_controller import regression_logistic_controller

app = Flask('flower-predict')
app.register_blueprint(regression_logistic_controller)

if __name__ == '__main__':
    app.run(debug=True, port=8000)