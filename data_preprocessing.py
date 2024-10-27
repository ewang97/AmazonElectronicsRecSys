import pandas as pd
import numpy as np
import argparse


def data_preprocess(data = 'ratings_Electronics.csv', n_ratings = 10):
    df = pd.read_csv(data)
    df = df.iloc[:1500000,0:]
    df.columns=['userId','productId','rating','timestamp']

    df.drop('timestamp',axis=1, inplace=True)
    df.drop_duplicates(inplace=True)
    print(f"\n Duplicates dropped.")
    
    prod_grp = df.groupby("productId").filter(lambda x:x['rating'].count() >= n_ratings)
    user_grp = df.groupby("userId").filter(lambda x:x['rating'].count() >= n_ratings)
    user_grp.drop(['rating'],inplace=True,axis=1)
    user_prod = pd.merge(prod_grp,user_grp)

    print(f"\n Data subsetted for users and items with more than ", n_ratings, " ratings.")

    print(f"\n Count of ratings remaining: " + str(user_prod.shape))

    user_prod.to_csv("cleaned_df_" + str(n_ratings) + ".csv", index=False)

    return user_prod


if __name__ == "__main__":
    data_preprocess(n_ratings = 20)
    
