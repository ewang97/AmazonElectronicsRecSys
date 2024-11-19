from sklearn import svm
from sklearn import datasets
import pickle


def save_model(model):
    with open('model.pkl','wb') as f:
        pickle.dump(model,f)


def load_model(model_pkl = 'model.pkl'):
    with open(model_pkl, 'rb') as f:
        latest_model = pickle.load(f)

    return latest_model