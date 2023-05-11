from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
from sqlalchemy import create_engine
import swifter
import re
import langdetect as ld
import psycopg2

def import_data():
    conn_string = 'postgresql://postgres:123@127.0.0.1/postgres'
    db = create_engine(conn_string)
    conn = db.connect()
    df = pd.read_sql_query('SELECT * FROM reddit_data_raw', con = conn)

    return df

def load_sentiment_data(df):
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

    sql = '''CREATE TABLE IF NOT EXISTS reddit_data_sentiments(index varchar, id varchar, 
                                                                text varchar, cleaned_text varchar, 
                                                                sentiment varchar);'''
    cursor.execute(sql)
    df.to_sql('reddit_data_sentiments', conn, if_exists = 'append')
    
    conn1.commit()
    conn1.close()
    
def remove_general(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  
        u"\U0001F300-\U0001F5FF"  
        u"\U0001F680-\U0001F6FF"  
        u"\U0001F1E0-\U0001F1FF"  
        u"\U00002700-\U000027BF"  
                            "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)
    text = re.sub('\S*@\S*\s?', '', text)
    text = re.sub('\s+', ' ', text)
    text = re.sub("\w+…|…", "", text) 
    text = re.sub("(?<=\w)-(?=\w)", "", text)
    text = re.sub("(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub("'", "", text)
    return text

def detect_english(text):
    try:
        return ld.detect(text) == 'en'
    except:
        return False
    
def sentiment_label(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest').to(device)
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    result = model(tokens)
    
    return int(torch.argmax(result.logits))


def main():
    df = import_data()
    df['cleaned_text'] = df['text'].swifter.apply(lambda x: remove_general(x))
    df['eng'] = df['cleaned_text'].swifter.apply(lambda x: detect_english(x))
    indexNotEng = df[(df['eng'] != True)].index
    df = df.drop(indexNotEng)
    df = df.drop('eng', axis=1)
    df['sentiment'] = df['cleaned_text'].swifter.apply(lambda x: detect_english(x))
    df.loc[df["sentiment"] == 0, "sentiment"] = "Negative"
    df.loc[df["sentiment"] == 1, "sentiment"] = "Neutral"
    df.loc[df["sentiment"] == 2, "sentiment"] = "Positive"
    
    df.to_csv('testsentiment.csv')
    df.reset_index(level=2, drop=True)
    load_sentiment_data(df)
    
if __name__ == "__main__":
    main()
    