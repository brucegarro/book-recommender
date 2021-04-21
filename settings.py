import os

DATA_PATH = os.path.abspath(os.path.join("..", "goodreads-dataset"))
BOOK_PATH = os.path.join(DATA_PATH, "book")
RATING_PATH = os.path.join(DATA_PATH, "user_rating")

BOOK_FILES = ([
    os.path.join(BOOK_PATH, name) for name in os.listdir(BOOK_PATH)
    if name.startswith("book")
])
RATING_FILES = ([
    os.path.join(RATING_PATH, name) for name in os.listdir(RATING_PATH)
    if name.startswith("user_rating")
])

ALL_BOOKS_FILE = os.path.join(BOOK_PATH, "all_books.csv")
ALL_RATINGS_FILE = os.path.join(RATING_PATH, "all_user_ratings.csv")
