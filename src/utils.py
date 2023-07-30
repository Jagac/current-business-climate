import re

import langdetect as ld
import pandas as pd
import psycopg2
from sqlalchemy import create_engine


def remove_general(text: str) -> str:
    """Preprocessing function that removes links emojis and other general

    Args:
        text (str): sentence/paragraph to be cleaned

    Returns:
        str: cleaned up text
    """
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r"", text)
    text = re.sub("\S*@\S*\s?", "", text)
    text = re.sub("\s+", " ", text)
    text = re.sub("\w+…|…", "", text)
    text = re.sub("(?<=\w)-(?=\w)", "", text)
    text = re.sub("(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub("'", "", text)
    return text


def detect_english(text: str) -> str | bool:
    """Some models preform better on English so this fucntion can be used to drop
    non English text.

    Args:
        text (str): sentence/paragraph to be cleaned

    Returns:
        boolean: tag True if english, False in not
    """
    try:
        return ld.detect(text) == "en"
    except:
        return False


def append_tables():
    conn1 = psycopg2.connect(
        database="postgres",
        user="postgres",
        password="123",
        host="127.0.0.1",
        port="5432",
    )

    conn1.autocommit = True
    cursor = conn1.cursor()

    sql = """
            CREATE TABLE IF NOT EXISTS combined_data 
            AS
             SELECT * FROM reddit_data
              UNION ALL
             SELECT * FROM youtube_data;"""

    cursor.execute(sql)
    conn1.commit()
    conn1.close()


def load_data() -> pd.DataFrame:
    """Loads a combined table of both sources

    Returns:
        pd.DataFrame: dataframe containing merged data
    """
    append_tables()
    conn_string = "postgresql://postgres:123@127.0.0.1/postgres"
    db = create_engine(conn_string)
    conn = db.connect()
    df = pd.read_sql_query("SELECT * FROM combined_data;", con=conn)
    print(f"--- Loaded {df.shape} ---")

    return df
