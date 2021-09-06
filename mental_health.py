# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 17:59:23 2021

@author: abhij
"""

import pandas as pd
import numpy as np

ds1=pd.read_csv('mh.csv')
ds2=pd.read_csv('mh2.csv')

x=ds2.iloc[0:1,2].values
t=pd.DataFrame({"A": ds2.iloc[0:1,0].values, "B":ds2.iloc[0:1,1] }).to_numpy()

for i in range(1,1855):
    if ds2.iloc[i,1]<ds2.iloc[i+1,1]:
        x=np.append(x,ds2.iloc[i+1:i+2,2].values)
        row=[ds2.iloc[i+1,0],ds2.iloc[i+1,1]]
        t=np.vstack([t,row])

X=ds1.iloc[:,1].values
X=np.append(X,x)

df=pd.DataFrame({"Questions": X})

from nltk.tokenize import RegexpTokenizer
tokenizer = RegexpTokenizer(r'\w+')
df['tokenized_words']= df.Questions.map(lambda t: tokenizer.tokenize(t))

#from nltk.stem import PorterStemmer
#ps=PorterStemmer()
#df['stemmed_words'] = df['tokenized_words'].map(
#    lambda l: [ps.stem(word) for word in l])

from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))
df['stopwords'] = df['tokenized_words'].map(
    lambda l:[word.lower() for word in l if not word.lower() in stop_words])

from nltk.stem import WordNetLemmatizer
lt=WordNetLemmatizer()
df['lemmetized_words'] = df['stopwords'].map(
    lambda l: [lt.lemmatize(word) for word in l])


df['string']=df['lemmetized_words'].map(lambda s: ' '.join(s))
X_train= df.iloc[:,4].values
X_train_copy= df.iloc[:,4].values


from sklearn.feature_extraction.text import CountVectorizer  
cv= CountVectorizer()
cv.fit(X_train)
X_train=cv.transform(X_train)

vocab=cv.vocabulary_
voc = cv.get_feature_names()
word_index = dict(zip(voc, range(len(voc))))

path_to_glove_file = 'glove.6B.100d.txt'


embeddings_dict = {}
with open(path_to_glove_file,encoding="utf8") as f:
    for line in f:
        word, coefs = line.split(maxsplit=1)
        coefs = np.fromstring(coefs, "f", sep=" ")
        embeddings_dict[word] = coefs

def closest_embeddings(embedding):
    return sorted(embeddings_dict.keys(), key=lambda word: np.linalg.norm(embeddings_dict[word]-embedding))


num_tokens = len(voc)*5
hits = 0
embedding_matrix = {}
for word, i in word_index.items():
        embedding_vector = embeddings_dict.get(word)
        
        if embedding_vector is not None:        
            embedding_matrix[word] = embedding_vector
            hits+=1
            list= closest_embeddings(embeddings_dict[word])[1:5] 
            for wrd in list:
                embedding_vector = embeddings_dict.get(wrd)     
                embedding_matrix[wrd] = embedding_vector
                hits+=1
        print(hits)
print("Converted %d words" % (hits))

np.save('embedding_matrix.npy', embedding_matrix)

embedding_matrix=np.load('embedding_matrix.npy')

words_list = list(embedding_matrix.keys())
words_list=np.array(words_list)

cv2=CountVectorizer()
cv2.fit(words_list)
X_train_copy=cv2.transform(X_train_copy)

pred_dict=np.zeros([912,100])
for i in range(911):
    vec=np.nonzero(X_train_copy[i,:])
    for j in vec[1]:
        pred_dict[i]=np.add(pred_dict[i], embedding_matrix[words_list[j]])
    if vec[1].size!=0:
        np.divide(pred_dict[i],vec[1].size)


from sklearn.metrics.pairwise import cosine_similarity

def chat():
    print("\nHello how can I help you(To leave the conversation type quit or bye)")
    while True:
        inp=input("User: ")
        print()
        if inp.lower() in ["hi", "hey", "is anyone there", "hello", "hay"]:
            arr = np.array(["Hello", "Hi", "Hi there"])
            print("Bot: ",np.random.choice(arr))
            continue
        elif inp.lower() in ["bye", "see you later", "goodbye","quit"]:
            arr2 = np.array(["see you later", "have a nice day", "bye! come back again"])
            print("Bot: ",np.random.choice(arr2))
            break
        elif inp.lower() in ["thanks", "thank you", "that's helpful", "thanks for the help"]:
            arr3 = np.array(["Thanks", "Thank you", "That's helpful", "Thanks for the help"])
            print("Bot: ",np.random.choice(arr3))
            continue
        elif inp.lower() in ["who are you", "what are you", "who you are" ]:
            arr4 = np.array(["I'm your bot assistant", "I'm an Artificial Intelligent bot"])
            print("Bot: ",np.random.choice(arr4))
            continue
        elif inp.lower() in ["could you help me", "give me a hand please", "can you help", "what can you do for me", "i need a support", "i need a help", "support me please"]:
            arr5 = np.array(["Tell me how can assist you", "Tell me your problem to assist you", "Yes Sure, How can I support you"])
            print("Bot: ",np.random.choice(arr5))
            continue
        inp=tokenizer.tokenize(inp)
        inpt=[]
        inp = [w.lower() for w in inp if not w.lower() in stop_words]
        for wrd in inp:
           inpt.append(lt.lemmatize(wrd)) 
        str=" "
        str=str.join(inpt)
        data=cv2.transform([str])
        ct=np.nonzero(data)
        pre=np.zeros((1,100))
        for k in ct[1]:
            pre[0]=np.add(pre[0], embedding_matrix[words_list[k]])
        if ct[1].size!=0:
            pre[0]=np.divide(pre[0],ct[1].size)
        max=0
        index=-1
        pre.reshape(-1,1)
        for i in range(911):
            sim=cosine_similarity(pre[0,:].reshape(1,-1),pred_dict[i,:].reshape(1,-1))
            if sim>max:
                max=sim
                index=i
        if index<98:
            print("Bot: ", ds1.iloc[index,2])
        else:
            num=index-98
            ds3=ds2.iloc[t[num,0]:t[num+1,0],8].values
            print("Bot: ",np.random.choice(ds3))
        
chat()


