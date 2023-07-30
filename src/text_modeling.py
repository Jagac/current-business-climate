import gensim.corpora as corpora
import optuna
import pandas as pd
import swifter
import torch
from bertopic import BERTopic
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sentence_transformers import SentenceTransformer
from sqlalchemy import create_engine
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from umap import UMAP

from utils import append_tables, load_data

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
# model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest').to(device)


def sentiment_label(text: str) -> int:
    """Uses a pretrained model from HuggingFace to label sentiments

    Args:
        text (str): text to be labeled

    Returns:
        int : 0 -> neg, 1 -> neut, 2 -> pos
    """
    encoded_input = tokenizer(text, padding=True, truncation=True)
    sentiment_model = torch.compile(model)
    tokens = tokenizer.encode(encoded_input, return_tensors="pt").to(device)
    result = model(tokens)

    return int(torch.argmax(result.logits))


def apply_sentiment_model(df: pd.DataFrame) -> pd.DataFrame:
    """Applies the sentiment label function

    Args:
        df (pd.DataFrame): table containing text

    Returns:
        pd.Dataframe: same table with an additional sentiment column with associated label
    """
    df2 = df.copy()
    print("--- Applying roBERTa ---")
    df2["sentiment"] = df2["cleaned_text"].swifter.apply(lambda x: sentiment_label(x))
    df2.loc[df2["sentiment"] == 0, "sentiment"] = "Negative"
    df2.loc[df2["sentiment"] == 1, "sentiment"] = "Neutral"
    df2.loc[df2["sentiment"] == 2, "sentiment"] = "Positive"

    return df2


def objective(trial: optuna.trial) -> float:
    """Bertopic is senstive to hyperparameters. Therefore function searches for best ones.

    Returns:
        float : coherence score i.e. how easy it is to interpret
    """
    params = {
        "n_neighbors": trial.suggest_int("n_neighbors", 5, 50),
        "n_components": trial.suggest_int("n_components", 5, 50),
        "min_cluster_size": trial.suggest_int("min_cluster_size", 5, 50),
        "min_samples": trial.suggest_int("min_samples", 5, 50),
    }
    umap_model = UMAP(
        n_neighbors=params["n_neighbors"],
        n_components=params["n_components"],
        metric="cosine",
        low_memory=False,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean",
        prediction_data=True,
    )

    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=False
    )
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
        [words for words, _ in topic_model.get_topic(topic) if words != ""]
        for topic in range(len(set(topics)) - 1)
    ]

    coherence_model = CoherenceModel(
        topics=topic_words,
        texts=tokens,
        corpus=corpus,
        dictionary=dictionary,
        coherence="u_mass",
    )
    coherence = coherence_model.get_coherence()

    return coherence


def tune_topic_model(tune=False):
    """Controls wether we want to start tuning or not. Usually takes some time and
    can be memory intensive.

    Args:
        tune (bool, optional): start tuning. Defaults to False.
    """
    if tune == True:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25)

        print("Best trial:")
        trial = study.best_trial
        print("Value: {}".format(trial.value))
        print("Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


def apply_topic_model(
    df: pd.DataFrame,
    n_neighbors: int,
    n_components: int,
    min_cluster_size: int,
    min_samples: int,
) -> pd.DataFrame:
    """Set hyperparams based on the tuning results. Have to manually assign a topic name using get_topic_info later"""
    df2 = df.copy()
    umap_model = UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        metric="cosine",
        low_memory=False,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )

    sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = sentence_model.encode(docs, show_progress_bar=True)
    topic_model = BERTopic(
        umap_model=umap_model, hdbscan_model=hdbscan_model, verbose=True
    ).fit(docs, embeddings)

    topic_model.reduce_topics(docs, nr_topics=10)
    topics, probs = topic_model.transform(docs, embeddings)

    df2 = pd.DataFrame({"docs": docs, "topics": topics})
    print(topic_model.get_topic_info())
    print(df2.shape)

    return df2


if __name__ == "__main__":
    df = load_data()
    df_sentiment = apply_sentiment_model(df)
    docs = df.cleaned_text.to_list()
    tune_topic_model(tune=False)
    topics_df = apply_topic_model(df, 42, 19, 49, 46)
