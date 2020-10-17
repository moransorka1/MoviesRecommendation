import numpy as np
import pandas as pd
import utils
from models import KNNMean, SVD_
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter('ignore')


DATA_PATH = './Data'

if __name__ == '__main__':
    # row data:
    ratings_df, movies_df = utils.load_and_cut_data(DATA_PATH)
    # join data:
    data = pd.merge(ratings_df, movies_df, on='movieId')

    data, movies_data = utils.data_preparation(data)
    utils.data_exploration(data, movies_data)

    # Q1
    top_10 = utils.get_top_movies(movies_data, 10)

    # Q2
    # Sample Data:
    data_sample = data.sample(frac=.2, replace=False, random_state=1)
    train, test = train_test_split(data_sample, test_size=0.4, random_state=1)

    # train, test = train_test_split(data, test_size=0.4, random_state=1)

    rating_scale = (data['rating'].min(), data['rating'].max())

    # Q3:
    knn = KNNMean(train, rating_scale)
    # knn.grid_search()

    test['knn_rating'] = knn.predict(test)
    knn_rmse = utils.rmse(test['knn_rating'], test['rating'])
    print(f'KNNWithMeans RMSE: {knn_rmse}')

    # SVD:
    SVD_grid_params = {"n_epochs": [50, 100],
                       "lr_all": [0.003, 0.005],
                       "reg_all": [0.01, 0.02]}
    svd = SVD_(train, rating_scale)
    # svd.grid_search(SVD_grid_params)

    test['svm_rating'] = svd.predict(test)
    svd_rmse = utils.rmse(test['svm_rating'], test['rating'])
    print(f'SVD RMSE: {svd_rmse}')


    # choose best:
    if svd_rmse < knn_rmse:
        best = 'SVD'
        test['pred_rating'] = test['svm_rating']
    else:
        best = 'KNN'
        test['pred_rating'] = test['knn_rating']

    print(f'Predict ratings using {best} model...')
    test.sort_values(by=['userId', 'pred_rating'], ascending=False, inplace=True)
    users_top10 = test.groupby('userId').head(10).loc[:, ['userId', 'movieId', 'title', 'rating', 'pred_rating', 'genres']]
    utils.print_df_as_table(users_top10.head(30))

