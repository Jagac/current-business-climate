from googleapiclient.discovery import build
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
from tqdm import tqdm
import os
from dotenv import load_dotenv
load_dotenv()

def extract_youtube_data(video_id):
    youtube = build('youtube','v3',
                    developerKey=os.getenv('youtube_api_key'))

    video_response=youtube.commentThreads().list(
      part='snippet, replies',
      videoId=f"{video_id}"
    ).execute()
    
    all_comments = []
    while video_response:
        for i, item in enumerate(video_response["items"]):
            comment = item["snippet"]["topLevelComment"]
            text = comment["snippet"]["textDisplay"]
            all_comments.append([text])
          
        if 'nextPageToken' in video_response:
            video_response = youtube.commentThreads().list(
                    part = 'snippet,replies',
                    videoId=f"{video_id}"
                ).execute()
        else:
            break
      
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

    sql = '''CREATE TABLE IF NOT EXISTS youtube_data_raw(index varchar, text varchar);'''
    cursor.execute(sql)
    df.to_sql('youtube_data_raw', conn, if_exists = 'append')
    
    conn1.commit()
    conn1.close()


def main():
    videos = ['Vp0LD9jgeV8', 'FgzyLoSkL5k', 'Qa_4c9zrxf0', 'RAjZ8EGuqV4', 
              'jLJTVPKrIH0', 'MdKuJtEJT2A']
    for video in tqdm(videos):
        df = extract_youtube_data(video)
        load_youtube_data(df)
        
if __name__ == '__main__':
    main()
  
  
 