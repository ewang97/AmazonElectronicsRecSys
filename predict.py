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

from util import save_model,load_model

def popularity_recommend(cleaned_csv_file = 'cleaned_df_20.csv', n = 5):
    df = pd.read_csv(cleaned_csv_file)
    df.columns=['userId','productId','rating']
    train_data, test_data = train_test_split(df, test_size = 0.3, random_state=5)

    #Count of user_id for each unique product as recommendation score 
    train_data_grouped = train_data.groupby('productId').agg({'userId': 'count'}).reset_index()
    train_data_grouped.rename(columns = {'userId': 'score'},inplace=True)
    
    #Sort the products on recommendation score 
    train_data_sort = train_data_grouped.sort_values(['score', 'productId'], ascending = [0,1]) 
    train_data_sort['rank'] = train_data_sort['score'].rank(ascending=0, method='first') 

    #Get the top n recommendations 
    popularity_recommendations = train_data_sort.head(n) 
    return popularity_recommendations 

def get_top_n(predictions, n=5):

    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r,est,_ in predictions:
        top_n[uid].append((iid))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]

    return top_n

def predict(cleaned_csv_file = 'cleaned_df_20.csv',latest_model = False):

    df = pd.read_csv(cleaned_csv_file)
    df.columns=['userId','productId','rating']
    reader = Reader(rating_scale=(1,5)) # rating scale range
    data = Dataset.load_from_df(df[['userId', 'productId', 'rating']], reader)
    # train_df, test_df = surprise_train_test_split(data, test_size=0.20)
    trainset = data.build_full_trainset()
    if latest_model:
        model = SVD()
        print("Fitting SVD...")
        model.fit(trainset)

        save_model(model)
    else:
        model = load_model('model.pkl')
        print("Fitting SVDpp...")
        model.fit(trainset)
        
    testset = trainset.build_anti_testset()

    predictions = model.test(testset)
    return predictions

if __name__ == "__main__":
    
    user_id = input("Enter a user id: ")

    print("Generating predictions...")
    predictions = predict()

    print("Getting top n...")
    top_n = get_top_n(predictions, n=5)

    #Predict top n product_ids for provided user_id
    if user_id in top_n:
        print(top_n[user_id])
    else:
        num_recs = input("User ID not found in dataset - please provide the number of popular products you would like recommended based on popularity: ")
        pop_recs = popularity_recommend(n = num_recs)
        print('Top ',num_recs,' Product Ids')
        print(pop_recs)

    #Sample ID: A25RTRAPQAJBDJ

    
