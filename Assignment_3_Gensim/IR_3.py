from nltk.stem.porter import PorterStemmer
import gensim
import pickle
from pathlib import Path
import numpy

import string
import codecs



def get_file_as_array(path):
    #1st return item is paragraphs as array that is stemmed, has stopwords
    #removed, and in lower case, without "gutenberg"
    #2nd return item is  paragraphs  as array indentical to text input
    f = codecs.open(path,"r","utf-8")
    translator = str.maketrans('','',string.punctuation)
    array = []
    T = []
    P=[]
    p=[]
    for line in f:
        l = line.lower().split()
        if l:
            P.extend(line.split())
            p.extend(l)
        elif p: #line is empty, and paragraph contains something
            for w in p:
                if "gutenberg" in w: #if gutenberg delete paragraph
                    p=[]
                    break;
            if p:
                #for i in range(len(p)):
                    #w= p[i].translate(translator) #removes symbols
                array.append(p) #adds paragraph to array
                p=[];
                T.append(P)            
                P=[]
        else: #the paragraph too is empty, implying to empty rows in a row
            p=[];
            P=[]
    return array, T

def remove_symbols(a):
    translator = str.maketrans('','',string.punctuation)
    A = []
    if any(isinstance(i,list) for i in a): #if two dimensional lsit
        i=0;
        for p in a:
            A.append([])
            for w in p:
                A[i].append(w.translate(translator))
            i+=1;
    else: #if one dimensional list
        for w in a:
            A.append(w.translate(translator))
    return A

def remove_stopwords(a,path = "stopwords.txt"):
    stopwords = get_stopwords_as_set(path)
    A = []
    i=0;
    if any(isinstance(i,list) for i in a): #if two dimensional lsit
        for p in a:
            A.append([])
            for w in p:
                if w not in stopwords:
                    A[i].append(w)
            i+=1;
    else:
        if w not in stopwords:
            A.append(w)
    return A;

def stem_array(a):
    stemmer = PorterStemmer()
    A = []
    i=0
    if any(isinstance(i,list) for i in a): #if two dimensional lsit
        for p in a:
            A.append([])
            for w in p:
                A[i].append(stemmer.stem(w)) #stems the word, updates word in p array
            i+=1
            
    else:
        for w in a:
            A.append(stemmer.stem(w))
    return a

def preprocessing(a,path = "stopwords.txt"):
    #removes symbols and stopwords, and stems the array
    #can handle both one and two dimensional array
    stopwords = get_stopwords_as_set(path)
    translator = str.maketrans('','',string.punctuation)
    stemmer = PorterStemmer()
    A = []
    i=0
    if any(isinstance(i,list) for i in a): #if two dimensional lsit
        for p in a:
            A.append([])
            for w in p:
                W = w.translate(translator) #remove symbols
                if W not in stopwords: #check if not stopwords
                    W = stemmer.stem(W) #stem the word
                    A[i].append(stemmer.stem(W)) #Add to paragraph
            i+=1
    else: #if one dimensional array
        r = "Remove words: "
        for w in a:
            W = w.translate(translator) #remove symbols
            if W not in stopwords: #check if not stopwords
                W = stemmer.stem(W) #stem the word
                A.append(stemmer.stem(W)) #Add to array
            else: r+= W + " "
        print(r)
    return A



def count_array(array):
    d = gensim.corpora.Dictionary(array)
    c = []
    for p in array:
        c.append(d.doc2bow(p))
    return c

def print_paragraph(p,maxl=100):
    s = ""
    for i in range(min(maxl,len(p))):
        s+=p[i] + " "
    if len(p)>maxl+2:
        s+="... \n---"+str(len(p)-maxl)+" words remaining in paragraph---"
    print(s)

def get_stopwords_as_set(path):
    f = codecs.open(path,"r","utf-8")
    stop = []
    for l in f:
        for w in l.split():
            stop.append(w)
    return set(stop)

def get_arrays(s,preload=True):
    print("Getting arrays")
    m = "binaries/"
    if preload and Path(m+s+"Preprocessed.p").is_file() and Path(m+s+"Text.p").is_file():
        with open(m+s+"Preprocessed.p", "rb") as f:
            a = pickle.load(f)
        with open(m+s+"Text.p", "rb") as f:
            raw_text = pickle.load(f)
    else:
        a,raw_text = get_file_as_array(s+".txt");
        #a = remove_symbols(a)
        #a = remove_stopwords(a)
        #a = stem_array(a)
        a = preprocessing(a)
        with open(m+s+"Preprocessed.p", "wb") as f:
            pickle.dump(a, f)
        with open(m+s+"Text.p", "wb") as f:
            pickle.dump(raw_text, f)
    return a,raw_text

def main():
    p = "pg3300"  #the entire file
    #lesser texts for faster processing
    p1 = "parTest"
    p2 = "pgTest"
    p3 = "queryTest"
    selected_path = p

    a,raw_text = get_arrays(selected_path,preload=True)
    print("Building dictionaries")
    dictionary = gensim.corpora.Dictionary(a)
    corpus = count_array(a) #corpus
    tfidf = gensim.models.TfidfModel(corpus) #tfidf_model from vectors
    corpus_tfidf = tfidf[corpus] #convert to real-valued weights
    index_tfidf = gensim.similarities.MatrixSimilarity(corpus)


    #PART 3.5
    print("\n---Part 3.5 Show result of 3 LSI topics")
    lsi = gensim.models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=100)
    corpus_lsi = lsi[corpus_tfidf]
    index_lsi = gensim.similarities.MatrixSimilarity(corpus_lsi)
    print("First 3 topics:")
    i=1
    for t in lsi.show_topics(3,num_words=5):
        print("Topic nr "+ str(i) + ":")
        print(t)
        i+=1
    #show_topics documentation ---- from radimurehurek.com:
    #Return num_topics most significant topics (return all by default). 
    #For each topic, show num_words most significant words (10 words by default).

    #PART 4.1 - 4.2
    print("\n---Part 4.2 Report TF-IDF weights---")
    string = "What is the function of Money"
    print("Result of: "+string)
    q = preprocessing(string.lower().split())
    query = dictionary.doc2bow(q)
    tfidf_query = tfidf[query]
    for i in range(len(q)):
        print(q[i] + " - " + str(tfidf_query[i]))
    print()
    input("Press ENTER to continoue...")
   
    #PART 4.3
    print("---Part 4.3 Find most relevant paragraphs from TFIDF model---")
    print("Query: " + string)
    d2s = enumerate(index_tfidf[tfidf_query]) 
    tfidf_result = sorted(d2s,key=lambda kv: -kv[1])[:3]
    print(tfidf_result)
    for i in tfidf_result:
        p = i[0]
        print("Paragraph nr: " + str(p))
        print_paragraph(raw_text[p])
        print()

    #PART 4.4
    print("\n---4.4 Find most relevant paragraphs from LSI model---")
    print("Query: " + string)
    lsi_query = lsi[tfidf_query]
    lsi_d2s = enumerate(index_lsi[lsi_query])
    lsi_result = sorted(lsi_d2s,key=lambda kv: -kv[1])[:3]
    for i in lsi_result:
        p = i[0]
        print("Paragraph nr: " + str(p))
        print_paragraph(raw_text[p])
        print()


if __name__== "__main__":
    main()
