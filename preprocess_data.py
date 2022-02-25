from nltk.corpus import stopwords
from bnlp.corpus import stopwords as bn_stopwords, punctuations, digits
from bnlp.corpus.util import remove_stopwords
import re
import string

import nltk
nltk.download('stopwords')


def preprocess_data_BN(text):
    #Stopwords, punc, digits
    text = ' '.join(remove_stopwords(text, bn_stopwords))
    text = ' '.join(remove_stopwords(text, punctuations))
    text = ' '.join(remove_stopwords(text, digits))

    # URL
    url = re.compile(r"https?://\S+|www\.\S+")
    text = url.sub(r"", text)

    # EMOJI
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    return text


def preprocess_data_EN_BN(text):
    stop = set(stopwords.words("english"))
    text = str(text)

    filtered_words = [word.lower()
                      for word in text.split() if word.lower() not in stop]
    text = " ".join(filtered_words)

    url = re.compile(r"https?://\S+|www\.\S+")
    text = url.sub(r"", text)

    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r'', text)

    return preprocess_data_BN(text)


df_processed = pd.read_csv('dataset.csv', na_values='nan')
df_processed.drop(columns=['Unnamed: 0'], inplace=True)

df_processed['text'] = df_processed.text.map(preprocess_data_EN_BN)
df_processed.to_csv('preprocessed_dataset.csv')
df_processed.text
