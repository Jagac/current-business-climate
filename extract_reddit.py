import praw
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm
import os
from utils import detect_english, remove_general
from dotenv import load_dotenv
load_dotenv()


def extract_reddit_data(subreddit_name):
    """Gets subreddit comments on hot posts ignoring the pinned ones

    Args:
        subreddit_name (str): name of subreddit not including "r/"

    Returns:
        pd.DataFrame: dataframe containing ids and comments
    """
    reddit = praw.Reddit(
                        client_id = os.getenv('reddit_client_id'), 
                        client_secret = os.getenv('reddit_client_secret'), 
                        user_agent = os.getenv('reddit_user_agent')
                        )

    all_comments = []
    subreddit = reddit.subreddit(subreddit_name)
    for post in subreddit.hot(limit=None):
        if not post.stickied:
            post.comments.replace_more(limit=0)
            comments = post.comments.list() 
            for comment in comments:
                all_comments.append([comment.id, comment.body])

    df = pd.DataFrame(all_comments, columns=['id', 'text'])

    return df

def load_reddit_data(df):
    """Load data to a raw table in postgres, appends every time so it can be rerun

    Args:
        df (pd.Dataframe): dataframe to be loaded
    """
    conn_string = 'postgresql://postgres:123@127.0.0.1/postgres'
    db = create_engine(conn_string)
    conn = db.connect()
    
    conn1 = psycopg2.connect(
        database="postgres",
        user='postgres', 
        password='123', 
        host='127.0.0.1', 
        port= '5432'
        )
    
    conn1.autocommit = True
    cursor = conn1.cursor()

    sql = '''CREATE TABLE IF NOT EXISTS reddit_data_raw(index varchar, id varchar, text varchar
                                                        cleaned_text varchar);'''
    cursor.execute(sql)
    df.to_sql('reddit_data_raw', conn, if_exists = 'append')
    
    conn1.commit()
    conn1.close()
      
      

    
    
def main():
    subreddit_names_list = ['startup', 'startups', 'smallbusiness', 'Business_Ideas']
    for subreddit in tqdm(subreddit_names_list):
        df = extract_reddit_data(subreddit)
        df['cleaned_text'] = df['text'].apply(lambda x: remove_general(x))
        df['eng'] = df['cleaned_text'].apply(lambda x: detect_english(x))
        indexNotEng = df[(df['eng'] != True)].index
        df = df.drop(indexNotEng)
        df = df.drop('eng', axis=1)
        
        load_reddit_data(df)
        
if __name__ == '__main__':
    main()
