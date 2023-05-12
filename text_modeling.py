from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import swifter
from utils import append_tables
append_tables()


def sentiment_label(text):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest')
    model = AutoModelForSequenceClassification.from_pretrained('cardiffnlp/twitter-roberta-base-sentiment-latest').to(device)
    sentiment_model = torch.compile(model) # torch 2.0 feature
    tokens = tokenizer.encode(text, return_tensors='pt').to(device)
    result = sentiment_model(tokens)
    
    return int(torch.argmax(result.logits))


def main():
    df['sentiment'] = df['cleaned_text'].swifter.apply(lambda x: sentiment_label(x))
    df.loc[df["sentiment"] == 0, "sentiment"] = "Negative"
    df.loc[df["sentiment"] == 1, "sentiment"] = "Neutral"
    df.loc[df["sentiment"] == 2, "sentiment"] = "Positive"
    
    
    
if __name__ == "__main__":
    main()
    