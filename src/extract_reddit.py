import os
from datetime import datetime

import pandas as pd
import praw
import psycopg2
from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm

from utils import detect_english, remove_general

load_dotenv()


def extract_reddit_data(subreddit_name: str) -> pd.DataFrame:
    """Gets subreddit comments on hot posts ignoring the pinned ones

    Args:
        subreddit_name (str): name of subreddit not including "r/"

    Returns:
        pd.DataFrame: dataframe containing ids and comments
    """
    reddit = praw.Reddit(
        client_id=os.getenv("reddit_client_id"),
        client_secret=os.getenv("reddit_client_secret"),
        user_agent=os.getenv("reddit_user_agent"),
    )

    all_comments = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=None):
        if not post.stickied:
            post.comments.replace_more(limit=0)
            comments = post.comments.list()
            for comment in comments:
                all_comments.append([comment.id, comment.body])

    df = pd.DataFrame(all_comments, columns=["id", "text"])

    return df


def load_reddit_data(df: pd.DataFrame) -> None:
    """Load data to a table in postgres, appends every time so it can be rerun

    Args:
        df (pd.Dataframe): dataframe to be loaded
    """
    conn_string = "postgresql://postgres:123@127.0.0.1/postgres"
    db = create_engine(conn_string)
    conn = db.connect()

    conn1 = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="123",
        host="127.0.0.1",
        port="5432",
    )

    conn1.autocommit = True
    cursor = conn1.cursor()

    sql = """CREATE TABLE IF NOT EXISTS reddit_data(index varchar, id varchar, last_refresh date,
                                                    text varchar, cleaned_text varchar);"""
    cursor.execute(sql)
    df.to_sql("reddit_data", conn, if_exists="append")

    conn1.commit()
    conn1.close()


def transform_reddit_data(df: pd.DataFrame) -> None:
    df2 = df.copy()
    df2["cleaned_text"] = df2["text"].apply(lambda x: remove_general(x))
    df2["eng"] = df2["cleaned_text"].apply(lambda x: detect_english(x))
    indexNotEng = df2[(df2["eng"] != True)].index
    df2 = df2.drop(indexNotEng)
    df2 = df2.drop("eng", axis=1)
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    df2["last_refresh"] = current_date
    df2 = df2[["id", "last_refresh", "text", "cleaned_text"]]

    return df2


def reddit_main() -> None:
    subreddit_names_list = ["startup", "startups", "smallbusiness", "Business_Ideas"]
    for subreddit in tqdm(subreddit_names_list):
        df = extract_reddit_data(subreddit)
        df_transformed = transform_reddit_data(df)
        load_reddit_data(df_transformed)


# if __name__ == '__main__':
#     reddit_main()
