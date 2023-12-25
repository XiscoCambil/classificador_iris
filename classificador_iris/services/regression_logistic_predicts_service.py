import pickle

def regression_logistic_predict(new_flower):

    with open('../models/iris_model.pck', 'rb') as f:
        loaded_model, loaded_sc  = pickle.load(f)

    new_flower_std = loaded_sc.transform(new_flower)

    prediction = loaded_model.predict(new_flower_std)

    return prediction