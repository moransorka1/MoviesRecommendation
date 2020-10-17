import numpy as np
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from ast import literal_eval
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

DATA_PATH = './Data'


def print_df_as_table(df):
    print(tabulate(df, headers='keys', tablefmt='psql'))

def dump_pickle(obj, file_path):
    print(f"dump to pickle {file_path}")
    with open(file_path, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_and_cut_data(data_path):
    try:
        ratings_df = pd.read_csv(f'{data_path}/ratings.csv', header=0)
        ratings_df['movieId'] = ratings_df['movieId'].astype('int64')

        movies_df = pd.read_csv(f'{data_path}/movies_metadata.csv', header=0)
        movies_df = movies_df.loc[:, ['id', 'title', 'genres']]
        movies_df.columns = ['movieId', 'title', 'genres']
        movies_df = movies_df[pd.to_numeric(movies_df['movieId'], errors='coerce').notnull()]
        movies_df['movieId'] = movies_df['movieId'].astype('int64')
        movies_df['genres'] = movies_df['genres'].fillna('[]').apply(literal_eval).apply(
            lambda x: ' '.join(i['name'].lower() for i in x) if isinstance(x, list) else '')

        # genres word2vec:
        # word2vec_model = load_word2vec_model()
        # movies_df['genres_embedding'] = movies_df['genres'].apply(
        #     lambda x: [word2vec_model[i] for i in x.split(' ') if i is not ''] if isinstance(x, str) else [])
        # movies_df['genres_embedding'] = movies_df['genres_embedding'].apply(lambda x: np.sum(x, axis=0))

        return ratings_df, movies_df
    except FileNotFoundError as e:
        print(e)


def data_preparation(data):
    data['timestamp'] = pd.to_datetime(data['timestamp'], unit='s')
    # calculate movies ratings:
    data['ratings_count'] = data.groupby('movieId')['rating'].transform('count')
    ratings_average = data.groupby('movieId')['rating'].mean().reset_index()
    ratings_average.columns = ['movieId', 'ratings_average']
    data = pd.merge(data, ratings_average, on='movieId')

    movies_data = data.loc[:,
                  ['movieId', 'title', 'genres', 'ratings_count', 'ratings_average']].drop_duplicates()

    return data, movies_data


def data_exploration(data, movies_data):
    print('------------------------')
    print('Ratings File Statistics:')
    print('------------------------')
    print(data.info())
    print('------------------------')
    movies = data['movieId'].nunique()
    print(f'Number of movies: {movies}')
    users = data['userId'].nunique()
    print(f'Number of users: {users}')
    print('------------------------')
    print('Top Rating Users:')
    top_rating_userts = pd.DataFrame(data.loc[:, 'userId'].value_counts().head(10)).reset_index()
    top_rating_userts.columns = ['userId', '# rated movies']
    print_df_as_table(top_rating_userts)
    print('------------------------')
    print('Top Rated Movies:')
    top_rated_movies = pd.DataFrame(data.loc[:, 'movieId'].value_counts().head(10)).reset_index()
    top_rated_movies.columns = ['movieId', '# rating users']
    print_df_as_table(top_rated_movies)

    plt.hist(data['rating'], bins=10)
    plt.title('Users Ratings Histogram')
    plt.xlabel('Rating')
    plt.ylabel('Number of observations')
    plt.show()
    # print(data['rating'].describe())

    average_ratings_per_movie = movies_data['ratings_average']
    plt.hist(average_ratings_per_movie)
    plt.title('Movies Average Ratings Histogram')
    plt.xlabel('Movie Average Rating')
    plt.ylabel('Number Of Movies')
    plt.show()
    # print(average_ratings_per_movie.describe())

    q_9 = movies_data['ratings_count'].dropna().quantile(0.9)
    ratings_count_per_movie = movies_data.loc[
        movies_data['ratings_count'] > q_9, 'ratings_count'].dropna().drop_duplicates()
    plt.hist(ratings_count_per_movie.dropna(), bins=100)
    plt.title('Number Of Ratings Per Movie')
    plt.xlabel('Number Of Ratings')
    plt.ylabel('Number Of Movies')
    plt.show()
    # print(movies_data['ratings_count'].describe())

    average_rating_per_user = data.groupby('userId')['rating'].mean()
    plt.hist(average_rating_per_user)
    plt.title('Users Average Rating Histogram')
    plt.xlabel('User Average Rating')
    plt.ylabel('Number Of Users')
    plt.show()
    # print(average_rating_per_user.describe())

    sns.jointplot(x='ratings_average', y='ratings_count', data=movies_data)
    plt.show()


def IMDB_weighted_rating(movie_data, m, C):
    """
    calculate IMDB's weighted rating for each movie
    :param movie_data: ratings_count and ratings_average is required
    :param m: minimum votes required
    :param C: the mean vote across the whole data
    v: number of votes for the movie
    R: the average rating of the movie
    :return: movie weighted rating
    """
    try:
        v = movie_data['ratings_count']
        R = movie_data['ratings_average']
        return (v / (v + m) * R) + (m / (m + v) * C)
    except Exception as e:
        print(e)


def get_top_movies(movies_data, top):
    """
    get top movies
    :param data: users rating df
    :param top: number of movies to return
    :return: top movies ranking by IMDB's weighted rating
    """
    # calculate minimum votes required:
    m = movies_data.loc[:, 'ratings_count'].dropna().quantile(0.9)
    # select movies with #votes greater than the calculated threshold:
    good_movies = movies_data.loc[(movies_data['ratings_count'] > m) & (movies_data['ratings_average'].notnull())]
    C = good_movies.loc[:, 'ratings_average'].dropna().mean()
    good_movies['wr'] = good_movies.loc[:, ['ratings_count', 'ratings_average']].apply(IMDB_weighted_rating,
                                                                                           args=(m, C,), axis=1)
    top_movies = good_movies.loc[:, ['title', 'ratings_count', 'ratings_average', 'wr', 'genres']].sort_values('wr', ascending=False).head(
        top)
    print(f'TOP {top} Movies:')
    print_df_as_table(top_movies)
    return top_movies


def rmse(pred, real):
    return np.sqrt(((pred - real) ** 2).mean())

def load_word2vec_model():
    word2vec_model = KeyedVectors.load_word2vec_format(f'{DATA_PATH}/GoogleNews-vectors-negative300.bin', binary=True)
    return word2vec_model

