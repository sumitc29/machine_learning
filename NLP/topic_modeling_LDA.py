import gensim

a=[]
for sentence in traindf['detokenized']:
    a.append(nltk.word_tokenize(sentence))

traindf['tokenized']=a

processed_docs=a

'''generate dictionary'''
dictionary = gensim.corpora.Dictionary(processed_docs)
len(dictionary)

count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 10:
        break

#filter dictionary having only more than 10 times occured words
dictionary.filter_extremes(no_below=10, no_above=0.1, keep_n= 100000)

len(dictionary)

bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

document_num = 120
bow_doc_x = bow_corpus[document_num]

for i in range(len(bow_doc_x)):
    print("Word {} (\"{}\") appears {} time.".format(bow_doc_x[i][0], 
                                                     dictionary[bow_doc_x[i][0]], 
                                                     bow_doc_x[i][1]))
    
lda_model =  gensim.models.LdaMulticore(bow_corpus, 
                                   num_topics = 8, 
                                   id2word = dictionary,                                    
                                   passes = 10,
                                   workers = 2)

for idx, topic in lda_model.print_topics(-1):
    print("Topic: {} \nWords: {}".format(idx, topic ))
    print("\n")
