from tensorflow import keras
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk

classNames = ['Profit', 'Loss']

Articles = []

Results = []

model = keras.Sequential([
    keras.layers.Flatten(50),
    keras.layers.Dense(50, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])