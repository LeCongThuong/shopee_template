import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


class TextProcessing:
    def preprocess_text(self, title):
        text = title.lower().strip()
        return text

    def __call__(self, title):
        return self.preprocess_text(title)

