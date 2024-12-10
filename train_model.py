import pandas as pd
import numpy as np
from surprise import Dataset, Reader, NormalPredictor, accuracy
from surprise.model_selection import cross_validate, train_test_split
from surprise.model_selection import GridSearchCV
from surprise import KNNBasic, KNNWithMeans, KNNWithZScore
from surprise import SVD, SVDpp, NMF
from surprise import SlopeOne, CoClustering

from util import save_model

def train_model(data = 'cleaned_df_10.csv'):
    reader = Reader(rating_scale=(1, 5))

    df = pd.read_csv(data)
    # Load data with rating scale

    dataset = Dataset.load_from_df(df,reader)

    # knnbasic_cv = cross_validate(KNNBasic(), dataset, cv=5, n_jobs=5, verbose=True)
    # knnmeans_cv = cross_validate(KNNWithMeans(), dataset, cv=5, n_jobs=5, verbose=True)
    # knnz_cv = cross_validate(KNNWithZScore(), dataset, cv=5, n_jobs=5, verbose=True)
    # svd_cv = cross_validate(SVD(), dataset, cv=5, n_jobs=5, verbose=True)
    svdpp_cv = cross_validate(SVDpp(), dataset, cv=5, n_jobs=5, verbose=True)
    # nmf_cv = cross_validate(NMF(), dataset, cv=5, n_jobs=5, verbose=True)
    # slope_cv = cross_validate(SlopeOne(), dataset, cv=5, n_jobs=5, verbose=True)
    # coclus_cv = cross_validate(CoClustering(), dataset, cv=5, n_jobs=5, verbose=True)

    print('SVDpp', '\t\t', round(svdpp_cv['test_rmse'].mean(), 4), '\t', round(svdpp_cv['test_mae'].mean(), 4))

    trainset, testset = train_test_split(dataset, test_size=0.25)

    svd_param_grid = {'n_epochs': [20, 25], 
                  'lr_all': [0.007, 0.009, 0.01],
                  'reg_all': [0.4, 0.6]}

    svdpp_gs = GridSearchCV(SVDpp, svd_param_grid, measures=['rmse', 'mae'], cv=5, n_jobs=5)
    svdpp_gs.fit(dataset)
    algo_svdpp = svdpp_gs.best_estimator['rmse']
              
    algo_svdpp.fit(trainset)
    # now test on the trainset            
    trainset_as_testset = trainset.build_testset()                                                      
    predictions_train = algo_svdpp.test(trainset_as_testset)                                           
    print('Accuracy on the trainset:')                                         
    accuracy.rmse(predictions_train)  

    # now test on the testset                                                  
                       
    pred_svdpp=algo_svdpp.test(testset)
    print('Accuracy on the testset:')                                          
    accuracy.rmse(pred_svdpp)  

    save_model(algo_svdpp)
    
    return pred_svdpp

if __name__ == "__main__":
    train_model()