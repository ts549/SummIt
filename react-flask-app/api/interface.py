import os
import speech_recognition as sr
import mutagen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import spacy
import pyLDAvis.gensim_models
pyLDAvis.enable_notebook()# Visualise inside a notebook
import en_core_web_md
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaMulticore
from gensim.models import CoherenceModel

class Interface:

    def data_collect(self, text, sentences):
        # Our spaCy model:
        nlp = en_core_web_md.load()
        # Tags I want to remove from the text
        removal= ['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE', 'NUM', 'SYM']
        tokens = []
        for summary in nlp.pipe(sentences):
            proj_tok = [token.lemma_.lower() for token in summary if token.pos_ not in removal and not token.is_stop and token.is_alpha]
            tokens.append(proj_tok)
        dictionary = Dictionary(tokens)
        dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=1000)
        corpus = [dictionary.doc2bow(doc) for doc in tokens]

        # Building the model
        lda_model = LdaMulticore(corpus=corpus, id2word=dictionary, iterations=50, num_topics=10, workers=4, passes=10)

    # def inverse_doc_freq(word, sentences, word_count):
    #     try:
    #         word_occurance = word_count[word] + 1
    #     except:
    #         word_occurance = 1
    #     return np.log(len(sentences)/word_occurance)

    # #Term Frequency
    # def termfreq(self, document, word):
    #     N = len(document)
    #     occurance = len([token for token in document if token == word])
    #     return occurance/N

    # def count_dict(self, sentences, word_set):
    #     word_count = {}
    #     for word in word_set:
    #         word_count[word] = 0
    #         for sent in sentences:
    #             if word in sent:
    #                 word_count[word] += 1

    #     return word_count

    def process_text(self, text):

        texts = [text]

        sentences = []
        word_set = []

        #Parse sentences and words
        for sent in texts:
            x = [i.lower() for i in word_tokenize(sent) if i.isalpha()]
            sentences.append(x)
            for word in x:
                if word not in word_set:
                    word_set.append(word)

        #Set of vocab
        word_set = set(word_set)
        #Total documents in our corpus
        total_documents = len(sentences)

        #Creating an index for each word in our vocab
        index_dict = {} #Dictionary to store index for each word
        i = 0
        for word in word_set:
            index_dict[word] = i
            i += 1

        word_count = count_dict(sentences, word_set)

        return

    def video_to_text(self, video):
        #Convert mp4 file to wav file
        command2mp3 = "ffmpeg -i Bolna.mp4 Bolna.mp3"
        command2wav = "ffmpeg -i Bolna.mp3 Bolna.wav"
        os.system(command2mp3)
        os.system(command2wav)

        r = sr.Recognizer()
        audio = sr.AudioFile(video)

        length = WAVE(audio).info.length

        #Transcribes the first 100 seconds of the video
        with audio as source:
            audio = r.record(source, duration=length*1000)
            text = r.recognize_google(audio)

        process_text(text)

        return text