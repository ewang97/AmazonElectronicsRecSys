import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from surprise import Dataset, Reader, SVD,NormalPredictor, accuracy
from surprise.model_selection import cross_validate
from surprise.model_selection import train_test_split as surprise_train_test_split

def get_top_n(predictions, n=5):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def predict(algo = 'svd'):

    df = pd.read_csv('ratings_Electronics.csv')
    df.columns=['userId','productId','rating','timestamp']
    reader = Reader(rating_scale=(0,5)) # rating scale range
    data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)

    if algo == 'svd':
        model = SVD()
    else:
        model = NormalPredictor()
    
    # train_df, test_df = surprise_train_test_split(data, test_size=0.20)
    trainset = data.build_full_trainset()
    model.fit(trainset)

    testset = trainset.build_anti_testset()

    predictions = model.test(testset)
    return predictions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recommend Top K Products for a user ID")
    
    predictions = predict(algo = 'svd')
    top_n = get_top_n(predictions, n=5)

    # Print the recommended items for each user
    for uid, user_ratings in top_n.items():
        print(uid, [iid for (iid, _) in user_ratings])