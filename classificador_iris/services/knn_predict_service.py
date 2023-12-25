import pickle

def knn_predict(new_flower):

    with open('../models/knn_model.pck', 'rb') as f:
        loaded_model, loaded_sc  = pickle.load(f)

    new_flower_std = loaded_sc.transform(new_flower)

    prediction = loaded_model.predict(new_flower_std)

    return prediction