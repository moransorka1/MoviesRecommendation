from surprise import Dataset
from surprise import Reader
from surprise import KNNWithMeans, SVD
from surprise.model_selection import GridSearchCV
from surprise.model_selection import cross_validate
import utils


class KNNMean:
    def __init__(self, data, rating_scale, k=50, min_k=1, sim_options=None):
        self.data = data
        self.rating_scale = rating_scale
        self.k = k
        self.min_k = min_k
        self.reader = Reader(rating_scale=self.rating_scale)
        if not sim_options:
            sim_options = {
                "name": "cosine",
                'min_support': 3,
                "user_based": False}  # Compute  similarities between items
        self.model_data = Dataset.load_from_df(data.loc[:, ["userId", "movieId", "rating"]], self.reader)
        self.trainset = self.model_data.build_full_trainset()
        self.model = KNNWithMeans(self.k, self.min_k, sim_options=sim_options)
        print('fitting KNNWithMeans model...')
        self.model.fit(self.trainset)
        self.grid_search_ = None

    def set_model_params(self, model_params):
        print('updating model parameters...')
        self.model = KNNWithMeans(model_params)
        print('fitting KNNWithMeans model...')
        self.model.fit(self.trainset)

    def update_grid_search(self, gs):
        self.grid_search_ = gs

    def fit(self, data):
        self.data = data
        self.model_data = Dataset.load_from_df(data.loc[:, ["userId", "movieId", "rating"]], self.reader)
        self.trainset = self.model_data.build_full_trainset()
        self.model.fit(self.trainset)

    def grid_search(self):
        print('grid search...')
        sim_options = {"name": ["msd", "cosine"],
                       "min_support": [3, 4],
                       "user_based": [False]}
        param_grid = {"sim_options": sim_options,
                      "k": [50, 100, 200],
                      "min_k": [1]}
        gs = GridSearchCV(KNNWithMeans, param_grid, measures=["rmse", "mae"], cv=3)
        gs.fit(self.model_data)
        best_params, best_score = gs.best_params["rmse"], gs.best_score["rmse"]
        print(f'Best score (RMSE): {best_score}')
        print(f'Best params (RMSE): {best_params}')

        print(f'Best score (MAE): {gs.best_score["mae"]}')
        print(f'Best params (RMSE): {gs.best_params["mae"]}')

        self.set_model_params(best_params)

        return best_params

    def predict(self, test_data):
        ratings = test_data.apply(lambda x: self.model.predict(x['userId'], x['movieId']).est, axis=1)
        return ratings


class SVD_:
    def __init__(self, data, rating_scale, n_epochs=50, lr_all=.005, reg_all=.02):
        self.data = data
        self.rating_scale = rating_scale
        self.reader = Reader(rating_scale=self.rating_scale)
        self.model_data = Dataset.load_from_df(data.loc[:, ["userId", "movieId", "rating"]], self.reader)
        self.trainset = self.model_data.build_full_trainset()
        self.model = SVD(n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
        print('fitting SVD model...')
        self.model.fit(self.trainset)
        self.grid_search_ = None

    def set_model_params(self, model_params):
        print('updating model parameters...')
        self.model = SVD(model_params)
        print('fitting SVD model...')
        self.model.fit(self.trainset)
        return self.model

    def update_grid_search(self, gs):
        self.grid_search_ = gs

    def fit(self, data):
        self.data = data
        self.model_data = Dataset.load_from_df(data.loc[:, ["userId", "movieId", "rating"]], self.reader)
        self.trainset = self.model_data.build_full_trainset()
        self.model.fit(self.trainset)

    def grid_search(self, grid_params):
        print('grid search...')
        gs = GridSearchCV(SVD, grid_params, measures=["rmse", "mae"], cv=3)
        gs.fit(self.model_data)
        best_params, best_score = gs.best_params["rmse"], gs.best_score["rmse"]
        print(f'Best score (RMSE): {best_score}')
        print(f'Best params (RMSE): {best_params}')

        print(f'Best score (MAE): {gs.best_score["mae"]}')
        print(f'Best params (RMSE): {gs.best_params["mae"]}')

        self.set_model_params(best_params)

        return best_params

    def predict(self, test_data):
        ratings = test_data.apply(lambda x: self.model.predict(x['userId'], x['movieId']).est, axis=1)
        return ratings
