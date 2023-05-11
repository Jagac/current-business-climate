import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import umap.umap_ as umap
import hdbscan
from hyperopt import fmin, tpe, hp, space_eval, Trials, partial
import spacy
import collections
from sqlalchemy import create_engine
from numba import jit

from numba.core.errors import NumbaDeprecationWarning, NumbaPendingDeprecationWarning
import warnings

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


def import_data():
    conn_string = 'postgresql://postgres:123@127.0.0.1/postgres'
    db = create_engine(conn_string)
    conn = db.connect()
    df = pd.read_sql_query('SELECT * FROM reddit_data_raw', con = conn)

    return df


def embed(text):
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print("module %s loaded" % module_url)
    
    return model(text)


def get_topics(embeddings, n_neighbors, n_components, min_cluster_size):
    dim_reduce = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components,
                           metric='cosine').fit_transform(embeddings)
    
    clusters = hdbscan.HDBSCAN(min_cluster_size = min_cluster_size,
                               metric='euclidean', cluster_selection_method='eom').fit(dim_reduce)

    return clusters


def cost_function(clusters, probability = 0.05):
    count = len(np.unique(clusters.labels_))
    total = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < probability)/total)
    
    return count, cost


def optimization_function(embeddings, params, lower_bound, upper_bound):
    clusters = get_topics(embeddings, n_neighbors = params['n_neighbors'], 
                          n_components = params['n_components'], 
                          min_cluster_size = params['min_cluster_size'])
    
    count, cost = cost_function(clusters, probability = 0.05)
    if (count < lower_bound) | (count > upper_bound):
        penalty = 0.15 
    else:
        penalty = 0
    loss = cost + penalty
    results = {'loss': loss, 'number of labels': count}

    return results


def optimization(embeddings, search_space, lower_bound, upper_bound, total_evaluations):
    trials = Trials()
    fmin_optimize = partial(optimization_function, embeddings = embeddings, 
                            lower_bound=lower_bound, upper_bound=upper_bound)
    
    best = fmin(fmin_optimize, search_space = search_space, algo=tpe.suggest, 
                total_evaluations = total_evaluations, trials=trials)
    
    best_params = space_eval(search_space, best)
    best_clusters = get_topics(embeddings, n_neighbors = best_params['n_neighbors'],
                               n_components = best_params['n_components'], 
                               min_cluster_size = best_params['min_cluster_size'])
    
    return best_params, best_clusters, trials


def get_group(df, label, category):
    group = df[df[label]==category]
    return group 

def most_common(list1, words):
    counter=collections.Counter(list1)
    most_frequent = counter.most_common(words)
    return most_frequent

def labels(docs):
    nlp = spacy.load("en_core_web_sm")
    
    verbs = [], object1 = [], nouns = [], adject = []
    verb = ''
    dobj = ''
    noun1 = ''
    noun2 = ''

    for i in range(len(docs)):
        doc = nlp(docs[i])

        for token in doc:
            if token.is_stop==False:
                if token.dep_ == 'ROOT':
                    verbs.append(token.text.lower())
                elif token.dep_=='OBJ':
                    object1.append(token.lemma_.lower())
                elif token.pos_=='NOUN':
                    nouns.append(token.lemma_.lower())
                elif token.pos_=='ADJ':
                    adject.append(token.lemma_.lower())
    
    if len(verbs) > 0:
        verb = most_common(verbs, 2)[0][0]
    if len(object1) > 0:
        object1 = most_common(object1, 1)[0][0]
    if len(nouns) > 0:
        noun1 = most_common(nouns, 2)[0][0]
    if len(set(nouns)) > 1:
        noun2 = most_common(nouns, 1)[1][0]

    labels = [verb, object1]
    
    for word in [noun1, noun2]:
        if word not in labels:
            labels.append(word)
    label = '_'.join(labels)
    
    return label


def summary(df, col):
    labels = df[col].unique()
    label_dict = {}
    for label in labels:
        current_label = list(get_group(df, col, label)['cluster_label'])
        label_dict[label] = labels(current_label)
        
    summary_df = (df.groupby(col)['cluster label'].count().sort_values('count', ascending=False))
    summary_df['topic'] = summary_df.swifter.apply(lambda x: label_dict[x[col]], axis = 1)
    
    return summary_df


def main():
    df = import_data()
    embeddings = embed(df['text'].values)
    
    
main()