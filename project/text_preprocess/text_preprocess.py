import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag, ne_chunk


class TextProcessing:

    def preprocess_text(self, title):
        """
            preprocess text
            sentence: input text sentence for preprocessing
        """
        # title = title.lower()
        title = title.translate(str.maketrans('', '', string.punctuation))
        title = title.strip()
        title = word_tokenize(title)
        title = [word for word in title if not word in stopwords.words('english')]
        title = ' '.join(title)
        return title

    def __call__(self, title):
        return self.preprocess_text(title)

