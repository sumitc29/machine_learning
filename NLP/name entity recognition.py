
'''using name entity recognition'''

'''using spacy'''
import spacy


s="Sumit is working in Amdocs and worked with TCS at Pune "

spacy_nlp = spacy.load('en')
document = spacy_nlp(s)

print('Original Sentence: %s' % (s))

for element in document.ents:
    print('Type: %s, Value: %s' % (element.label_, element))

    
'''using nltk'''
'''unable to get it in nltk try to find out solution'''
import nltk
tags=nltk.pos_tag(s.split())

nltk.ne_chunk(tags)

nltk.ne_chunk(tags).draw()