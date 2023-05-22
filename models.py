import pickle

with open('models/model_clf.pkl','rb') as f:
    clf = pickle.load(f)

with open('models/model_nb.pkl','rb') as f:
    model = pickle.load(f)

with open('models/model_neigh.pkl','rb') as f:
    neigh = pickle.load(f)

with open('models/model_svc.pkl','rb') as f:
    svc = pickle.load(f)

with open('models/model_tree.pkl','rb') as f:
    tree = pickle.load(f)