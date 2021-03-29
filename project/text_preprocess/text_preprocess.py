import string
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


class TextProcessing:
    def preprocess_text(self, title):
        text = title.lower().translate(str.maketrans('','', string.punctuation)).strip()
        text_wo_sw = [word for word in text if not word in stopwords.words()]
        return text_wo_sw

    def __call__(self, title):
        return self.preprocess_text(title)

