from googleapiclient.discovery import build
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm
import os
from datetime import datetime
from utils import detect_english, remove_general
from dotenv import load_dotenv
load_dotenv()

def extract_youtube_data(youtube, video_id, all_comments=[], token=''):
    video_response=youtube.commentThreads().list(part='snippet',
                                               videoId=video_id,
                                               pageToken=token).execute()
  
    for item in video_response['items']:
        comment = item['snippet']['topLevelComment']
        text = comment['snippet']['textDisplay']
        all_comments.append(text)
    
    if "nextPageToken" in video_response: 
        return extract_youtube_data(youtube, video_id, all_comments, video_response['nextPageToken'])
    else:
        df = pd.DataFrame(all_comments, columns=['text'])
        return df

def load_youtube_data(df):
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

    sql = '''CREATE TABLE IF NOT EXISTS youtube_data(index varchar, text varchar, 
                                                        cleaned_text varchar, last_refresh date);'''
    cursor.execute(sql)
    df.to_sql('youtube_data_raw', conn, if_exists = 'append')
    
    conn1.commit()
    conn1.close()

def transform_youtube_data(df, video_id):
    df['cleaned_text'] = df['text'].apply(lambda x: remove_general(x))
    df['eng'] = df['cleaned_text'].apply(lambda x: detect_english(x))
    indexNotEng = df[(df['eng'] != True)].index
    df = df.drop(indexNotEng)
    df = df.drop('eng', axis=1)
    now = datetime.now()
    current_date = now.strftime("%Y-%m-%d")
    df['last_refresh'] = current_date
    df['id'] = video_id
    df = df[['id', 'last_refresh', 'text', 'cleaned_text']]

    return df

def youtube_main():
    youtube = build('youtube', 'v3', developerKey=os.getenv('youtube_api_key'))
    videos = ['Vp0LD9jgeV8','Qa_4c9zrxf0', 'RAjZ8EGuqV4', 
              'jLJTVPKrIH0', 'MdKuJtEJT2A', 'FgzyLoSkL5k']
    
    for video in tqdm(videos):
        df = extract_youtube_data(youtube=youtube, video_id=video)
        df = transform_youtube_data(df, video_id=video)
        load_youtube_data(df)
        
        
if __name__ == '__main__':
    youtube_main()
  
  
 