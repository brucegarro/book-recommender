import io
import csv
import pandas as pd
import torch

from settings import (
    BOOK_FILES, RATING_FILES,
    ALL_BOOKS_FILE, ALL_RATINGS_FILE,
)
from matrix_factorization import matrix_factorization
from bruces_ratings import Bruces_ratings

BOOK_COLUMNS = [
    "Id",
    "Name",
    # "Author",
    "Rating",
]

def load_books_df():
    books_df = None
    for filename in BOOK_FILES:
        df = pd.read_csv(filename, usecols=BOOK_COLUMNS)
        if books_df is None:
            books_df = df
        else:
            books_df = books_df.merge(df, how="outer", on="Id")
    books_df = books_df[BOOK_COLUMNS]
    books_df = books_df.set_index("Id")
    return books_df

books_df = load_books_df()

RATING_COLUMNS = [
    "ID",
    "Name",
    "Rating",
]

RATING_MAP = {
    "really liked it": 5.0,
    "it was amazing": 4.0,
    "liked it": 3.0,
    "it was ok": 2.0,
    "did not like it": 1.0,
    "This user doesn't have any rating": 0.0,
}
def rating_to_val(rating_str):
    return RATING_MAP[rating_str]

def load_ratings_df(limit=None):
    ratings_df = None
    for filename in RATING_FILES:
        df = pd.read_csv(filename, usecols=RATING_COLUMNS)
        if ratings_df is None:
            ratings_df = df
        else:
            ratings_df = pd.concat([ratings_df, df])

    ratings_df = ratings_df.rename({"ID": "UserId"}, axis="columns")
    ratings_df["Rating"] = ratings_df["Rating"].apply(rating_to_val)

    if limit:
        rating_counts_df = ratings_df.groupby("Name").count().sort_values(by="Rating", ascending=False)
        rating_counts_df = rating_counts_df.head(limit+1)
        book_names = set(rating_counts_df.index[1:])

        # Include all books I've read
        book_names |= set(Bruces_ratings.keys())

        ratings_df = ratings_df[ratings_df["Name"].isin(book_names)]

    return ratings_df, rating_counts_df

# ratings_df, rating_counts_df = load_ratings_df(limit=10000)
ratings_df, rating_counts_df = load_ratings_df(limit=1000)

def create_book_indexes(ratings_df):
    book_index = { name: i for i, name in enumerate(ratings_df["Name"].unique()) }
    reverse_index = { i: name for name, i in book_index.items() }
    return book_index, reverse_index

book_index, reverse_index = create_book_indexes(ratings_df)

def show_found_books(book_index):
    for book_name, _ in Bruces_ratings.items():
        print(book_name if book_index.get(book_name) else "-_-", book_index.get(book_name, "Not found"))
show_found_books(book_index)

def append_self_ratings(ratings_df, book_index):
    self_df = pd.DataFrame(
        [{"UserId": 0, "Name": book_name, "Rating": rating} for book_name, rating in Bruces_ratings.items() ]
    , columns=ratings_df.columns)
    return ratings_df.append(self_df)
ratings_df = append_self_ratings(ratings_df, book_index)


def create_user_index(ratings_df):
    return { user_id: idx for idx, user_id in enumerate(sorted(ratings_df["UserId"].unique())) }
user_index = create_user_index(ratings_df)

def create_ratings_matrix(ratings_df, user_index, book_index):
    num_users, num_items = (len(user_index), len(book_index))
    ratings = torch.zeros((num_users, num_items))
    for _, rating in ratings_df.iterrows():
        user_i = user_index[rating["UserId"]]
        item_i = book_index[rating["Name"]]
        ratings[user_i][item_i] = rating["Rating"]
    return ratings

ratings = create_ratings_matrix(ratings_df, user_index, book_index)

lr = 0.005

P, Qt = matrix_factorization(R=ratings, K=2, steps=10, lr=lr)

# Predict and view
predictions = torch.matmul(P, Qt)
bruce_predictions = predictions[user_index[0]]

bruce_predictions = sorted([
    (reverse_index[i], float(score)) for i, score in enumerate(bruce_predictions)
], key=lambda x: x[2], reverse=True)

self_prediction_df = pd.DataFrame([{
    "rank": i+1,
    "title": title,
    "score": score,
} for i, (_, title, score) in enumerate(bruce_predictions) ])
