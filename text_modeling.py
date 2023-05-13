from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import swifter
#from utils import append_tables
from sqlalchemy import create_engine
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel
import optuna
from umap import UMAP
from hdbscan import HDBSCAN


def load_data():
    #append_tables()
    conn_string = 'postgresql://postgres:123@127.0.0.1/postgres'
    db = create_engine(conn_string)
    conn = db.connect()
    df = pd.read_sql_query("SELECT * FROM youtube_data;", con=conn)
    
    return df
    
    
def sentiment_label(text):
    """Uses a pretrained model from HuggingFace to label sentiments

    Args:
        text (str): _text to be labeled

    Returns:
        int : 0 -> neg, 1 -> neut, 2 -> pos
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest').to(device)
    sentiment_model = torch.compile(model) # torch 2.0 feature
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    result = sentiment_model(tokens)
    
    return int(torch.argmax(result.logits))


def sentiment_main(df):
    df2 = df.copy()
    df2['sentiment'] = df2['cleaned_text'].swifter.apply(lambda x: sentiment_label(x))
    df2.loc[df2["sentiment"] == 0, "sentiment"] = "Negative"
    df2.loc[df2["sentiment"] == 1, "sentiment"] = "Neutral"
    df2.loc[df2["sentiment"] == 2, "sentiment"] = "Positive"
    
    return df2
    


def objective(trial):
    params = {
        'n_neighbors':trial.suggest_int('n_neighbors', 5, 50),
        'n_components':trial.suggest_int('n_components', 5, 50),
        'min_cluster_size':trial.suggest_int('min_cluster_size', 5, 50),
        'min_samples':trial.suggest_int('min_samples', 5, 50)
    }
    umap_model = UMAP(n_neighbors= params['n_neighbors'], n_components=params['n_components'], 
                      metric='cosine', low_memory=False)
    hdbscan_model = HDBSCAN(min_cluster_size=params['min_cluster_size'], min_samples=params['min_samples'],
                            metric='euclidean', prediction_data=True)

    topic_model = BERTopic(umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False)
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    topics, _ = topic_model.fit_transform(docs)
    vectorizer = topic_model.vectorizer_model
    analyzer = vectorizer.build_analyzer()
    tokens = [analyzer(doc) for doc in docs]
    dictionary = corpora.Dictionary(tokens)
    corpus = [dictionary.doc2bow(token) for token in tokens]
    topics = topic_model.get_topics()
    topics.pop(-1, None)
    
    topic_words = [
        [words for words, _ in topic_model.get_topic(topic) if words!=''] 
               for topic in range(len(set(topics))-1)]

    coherence_model = CoherenceModel(topics=topic_words, 
                                texts=tokens, 
                                corpus=corpus,
                                dictionary=dictionary, 
                                coherence='u_mass')
    coherence = coherence_model.get_coherence()
    
    return coherence
    
def tune_topic_model():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=25)
    
    print("Best trial:")
    trial = study.best_trial
    print("Value: {}".format(trial.value))
    print("Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
if __name__ == "__main__":
    df = load_data()
    docs = df.cleaned_text.to_list()
    tune_topic_model()