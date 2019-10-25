import numpy as np
import re
from tqdm import tqdm
import scipy
import seaborn as sns

model = {}
sentences = {}

all_input_sentences = []

def loadGloveModel(glove_path, glove_dim):
    print("Loading Glove Model")

    vocab_size = int(4e5)  # this is the vocab size of the corpus we've downloaded

    # go through glove vecs
    with open(glove_path, 'r', encoding="utf8") as fh:
        for line in tqdm(fh, total=vocab_size):
            splitLine = line.split()
            word = splitLine[0]
            embedding = np.array([float(val) for val in splitLine[1:]])
            model[word] = embedding
    print ("Done.",len(model)," words loaded!")
    assert len(model) == vocab_size
    return model

def preprocess(raw_text):

    # keep only words
    letters_only_text = re.sub("[^a-zA-Z]", " ", raw_text)

    # convert to lower case and split
    words = letters_only_text.lower().split()

    return words

def cosine_distance_wordembedding_method(s1, s2):
    # if key in dict.keys(): model[word]
    vector_1 = np.mean([model[word] for word in preprocess(s1)],axis=0)
    vector_2 = np.mean([model[word] for word in preprocess(s2)],axis=0)
    cosine = scipy.spatial.distance.cosine(vector_1, vector_2)
    print('Sentence',s2,'is similar to gold standart for:',round((1-cosine)*100,2),'%')
    return cosine

def cosine_distance_between_two_words(word1, word2):
    return (1- scipy.spatial.distance.cosine(model[word1], model[word2]))

def load_sentence(sentence_path, gold_sentence_path):
    print("Loading sentences")

    # go through input sentence.txt list
    with open(sentence_path, 'r', encoding="utf8") as sen:
        for line in tqdm(sen):
            splitLine = line.strip().split(".")
            all_input_sentences.append(splitLine)

    # go through gold sentence.txt list
    with open(gold_sentence_path, 'r', encoding="utf8") as gs:
        counter = 0
        for line in tqdm(gs):
            sentences[line.strip()] = all_input_sentences[counter]
            counter = counter + 1
    return sentences

def comparison_sentences(sentence_path, gold_sentence_path):
    sentences = load_sentence(sentence_path, gold_sentence_path)

    for gold_sentence, all_input_sentences in sentences.items():
        print('Word Embedding method with a cosine distance between two sentences:')
        best_score = 0
        for input_sentence in all_input_sentences:
            cosine = round((1-cosine_distance_wordembedding_method(gold_sentence, input_sentence.strip()))*100,2)
            if cosine > best_score:
                best_score = cosine
                best_sentence = [gold_sentence, input_sentence.strip(), best_score]
        print('The most similar sentences:', best_sentence)
        print('============================')
