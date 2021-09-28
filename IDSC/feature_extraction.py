import sys
import ast
import re
from itertools import groupby
import numpy as np
import collections
from gensim.models import Word2Vec
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import logging
import pickle

#logging setup
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)

class feature_extractor(object):
    
    def __init__(self,txt_path,embedding_size=512):

        #store records
        labels = []
        tokens = []
        maxsentlen = 0
        maxdoclen = 0

        #process txt one line at a time
        with open(txt_path,'r') as f:
            lineno = 0
            for line in f:
            
                lineno += 1
                sys.stdout.write("processing line %i of approx 200,000     \r" \
                                 % lineno)
                sys.stdout.flush()
                line = f.readline().split("\t")
                
                text = line[2]
                        
                #process text
                text = text.lower()
                text = text.translate({ord(i): None for i in '()'})
                text = re.sub("dr.",'dr', text)
                text = re.sub('m.d.', 'md', text)
                text = re.sub('a.m.','am', text)
                text = re.sub('p.m.','pm', text)
                text = re.sub("\d+\.\d+", 'floattoken', text)
                text = re.sub("['=[]*&#]", '', text)
                text = re.sub("\.{2,}", '.', text)
                text = re.sub('[^\w_|\.|\?|!]+', ' ', text)
                text = re.sub('\.', ' . ', text)
                text = re.sub('\?', ' ? ', text)
                text = re.sub('!', ' ! ', text)
                text = re.sub('305 \d{3} \w{0,4}\d{4}', '', text)
                text = re.sub('\w{0,4}(phone)', '', text)
                text = re.sub('fax', '', text)
                text = re.sub('1400 nw \d* ?\w* ?room ?\d*', '', text)

                #tokenize
                text = text.split()
                
                #drop empty reviews
                if len(text) == 0:
                    continue

                #split into sentences
                sentences = []
                sentence = []
                for t in text:
                    if t not in ['.','!','?']:
                        sentence.append(t)
                    else:
                        sentence.append(t)
                        sentences.append(sentence)
                        if len(sentence) > maxsentlen:
                            maxsentlen = len(sentence)
                        sentence = []
                if len(sentence) > 0:
                    sentences.append(sentence)
                
                #add split sentences to tokens
                tokens.append(sentences)
                if len(sentences) > maxdoclen:
                    maxdoclen = len(sentences)
                    
                #add label 
                labels.append(line[1])
                
        print('\nsaved %i records' % len(tokens))
                
        #generate Word2Vec embeddings
        print("generating word2vec embeddings")

        #used all processed raw text to train word2vec
        self.allsents = [sent for doc in tokens for sent in doc]

        self.model = Word2Vec(sentences=self.allsents, vector_size=embedding_size, min_count=5, workers=4, epochs=5)
        # self.model.init_sims(replace=True) # this function is obsoleted and destructive, 
        #normed vectors are now generated dynamically on demand
        
        #save all word embeddings to matrix
        print("saving word vectors to matrix")
        self.vocab = np.zeros((len(self.model.wv)+1,embedding_size))
        word2id = {}

        #first row of embedding matrix isn't used so that 0 can be masked
        wv = self.model.wv
        for key in wv.index_to_key:
            idx = wv.get_index(key) + 1
            self.vocab[idx,:] = wv.get_vector(key, norm=True)
            word2id[key] = idx
            
        #normalize embeddings
        self.vocab -= self.vocab.mean()
        self.vocab /= (self.vocab.std()*2.5)

        #reset first row to 0
        self.vocab[0,:] = np.zeros((embedding_size))

        #add additional word embedding for unknown words
        self.vocab = np.concatenate((self.vocab, np.random.rand(1,embedding_size)))

        #index for unknown words
        unk = len(self.vocab)-1

        #convert words to word indicies
        print("converting words to indices")
        self.data = {}
        for idx,doc in enumerate(tokens):
            sys.stdout.write('processing %i of %i records       \r' % (idx+1,len(tokens)))
            sys.stdout.flush()
            dic = {}
            dic['label'] = labels[idx]
            dic['text'] = doc
            indicies = []
            for sent in doc:
                indicies.append([word2id[word] if word in word2id else unk for word in sent])
            dic['idx'] = indicies
            self.data[idx] = dic
    
    def visualize_embeddings(self):
        
        #get most common words
        print("getting common words")
        allwords = [word for sent in self.allsents for word in sent]
        counts = collections.Counter(allwords).most_common(500)

        #reduce embeddings to 2d using tsne
        print("reducing embeddings to 2D")
        embeddings = np.empty((500,embedding_size))
        for i in range(500):
            embeddings[i,:] = model[counts[i][0]]
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=7500)
        embeddings = tsne.fit_transform(embeddings)

        #plot embeddings
        print("plotting most common words")
        fig, ax = plt.subplots(figsize=(30, 30))
        for i in range(500):
            ax.scatter(embeddings[i,0],embeddings[i,1])
            ax.annotate(counts[i][0], (embeddings[i,0],embeddings[i,1]))
        plt.show()
        
    def get_embeddings(self):
        return self.vocab

    def get_data(self):
        return self.data
    
if __name__ == "__main__": 

    #get txt filepath
    args = (sys.argv)
    if len(args) != 2:
        raise Exception("Usage: python feature_extraction.py <path to txt file>")
    txt_path = args[1]
    
    #process txt
    fe = feature_extractor(txt_path,512)
    vocab = fe.get_embeddings()
    data = fe.get_data()
    
    #create directory for saved model
    if not os.path.exists('./data'):
        os.makedirs('./data')
    
    #save vocab matrix and processed documents
    np.save('./data/icd10_embeddings',vocab)
    with open('./data/icd10_data.pkl', 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
