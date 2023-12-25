import pickle

def decision_tree_predict(new_flower):

    with open('../models/decision_tree_model.pck', 'rb') as f:
        loaded_model, loaded_sc  = pickle.load(f)

    new_flower_std = loaded_sc.transform(new_flower)

    prediction = loaded_model.predict(new_flower_std)

    return prediction