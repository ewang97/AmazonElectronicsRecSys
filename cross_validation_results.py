import pandas as pd
import numpy as np
from surprise import Dataset, Reader, NormalPredictor, accuracy
from surprise.model_selection import cross_validate, train_test_split
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering

from util import save_model

def cv_results(data = 'cleaned_df_10.csv',model = 'SVDpp'):
    reader = Reader(rating_scale=(1, 5))

    df = pd.read_csv(data)
    # Load data with rating scale

    dataset = Dataset.load_from_df(df,reader)

    if model == 'SVDpp':
        cv = cross_validate(SVDpp(), dataset, cv=5, n_jobs=5, verbose=True)
    elif model == 'KNNBasic':
        cv = cross_validate(KNNBasic(), dataset, cv=5, n_jobs=5, verbose=True)
    elif model == 'KNNWithMeans':
        cv = cross_validate(KNNWithMeans(), dataset, cv=5, n_jobs=5, verbose=True)
    elif model == 'KNNWithZScore':
        cv = cross_validate(KNNWithZScore(), dataset, cv=5, n_jobs=5, verbose=True)
    elif model == 'SVD':
        cv = cross_validate(SVD(), dataset, cv=5, n_jobs=5, verbose=True)
    elif model == 'NMF':
        cv = cross_validate(NMF(), dataset, cv=5, n_jobs=5, verbose=True)
    elif model == 'SlopeOne':
        cv = cross_validate(SlopeOne(), dataset, cv=5, n_jobs=5, verbose=True)
    elif model == 'CoClustering':
        cv = cross_validate(CoClustering(), dataset, cv=5, n_jobs=5, verbose=True)
        
    else:
        print("Model not recognized. Please enter model supported by Surprise library.")
        return

    print('Algorithm\t RMSE\t MAE')
    print()
    print(model, '\t', round(cv['test_rmse'].mean(), 4), '\t', round(cv['test_mae'].mean(), 4))
    
    return

if __name__ == "__main__":
    cv_results(model = "SVD")
    cv_results(model = "SVDpp")
    # cv_results(model = "KNNBasic")
    # cv_results(model = "KNNWithMeans")
    # cv_results(model = "KNNWithZScore")
    cv_results(model = "NMF")