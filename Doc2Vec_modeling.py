import tensorflow as tf

# importing all necessary modules
# from nltk.tokenize import sent_tokenize, word_tokenize
import warnings

warnings.filterwarnings(action='ignore')

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt')

file = open('NewsArticles.txt', 'r')

sentences = file.read().replace('\n', ' ').split('. ')
# print(len(sentences))
# quit()



# for i in range(100):
#     sentences.append(file.readline().replace('\n', '').replace('.', '').replace(',', '').replace('!', '').replace('?', '').lower())
print(sentences)
tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(sentences)]
print(tagged_data)
max_epochs = 100
vec_size = 50
alpha = 0.025

model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm =1, workers=4)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save('ExampleDoc2VecModel')

test_data = word_tokenize("it makes profit".lower())
print("testdata:", test_data)
v1 = model.infer_vector(test_data)
# print("V1_infer", v1)

# to find most similar doc using tags
similar_doc = model.docvecs.most_similar(positive=[v1], topn=5)
print(sentences[int(similar_doc[0][0])], similar_doc)



# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
# print(model.docvecs['1'])

