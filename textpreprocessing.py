import re
import nltk

#1 Convert text to lowercase
input_str = 'The 5 biggest countries by population in 2017 are China, India, United States, Indonesia, and Brazil.'
input_str = input_str.lower()
#print(input_str)





#2  Remove numbers
input_str = 'Box A contains 3 red and 5 white balls, while Box B contains 4 red and 2 blue balls.'
#Remove numbers if they are not relevant to your analyses.
#Usually, regular expressions are used to remove numbers.
result = re.sub(r'\d+','', input_str)
#print(input_str)





#3 Remove punctuation
# define punctuation
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
my_str = "Hello!!!, he said ---and went."
# To take input from the user
# my_str = input("Enter a string: ")
# remove punctuation from the string
no_punct = ""
for char in my_str:
   if char not in punctuations:
       no_punct = no_punct + char
# display the unpunctuated string
#print(no_punct)





#4 Remove whitespaces
input_str = ' \t a string example\t '
input_str = input_str.strip()
#input_str




#-------------------------------------------------------------------------------------
# TOKENIZATION
#Tokenization is the process of splitting the given text into smaller pieces
#called tokens.
#Words, numbers, punctuation marks, and others can be considered as tokens.



# 1 Stop words removal
# "Stop Words"are the most common words in a language like
#“the”, “a”, “on”, “is”, “all”.
#These words do not carry important meaning and are usually removed from texts. 
import nltk
sentence = '''At eight o'clock on Thursday morning
... Arthur didn't feel very good.'''
tokens = nltk.word_tokenize(sentence)
#print(tokens)

#tagged = nltk.pos_tag(tokens)
#tagged[0:6]





# 2 Stemming is a process of reducing words
#to their word stem, base or root form
#(for example, books — book, looked — look).
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
stemmer= PorterStemmer()
input_str='There are several types of stemming algorithms.'
input_str=word_tokenize(input_str)
#for word in input_str:
#    print(stemmer.stem(word))





# 3 Lemmatization
# The aim of lemmatization, like stemming, is to reduce inflectional forms to a common base form.
# As opposed to stemming, lemmatization does not simply chop off inflections.
# Instead it uses lexical knowledge bases to get the correct base forms of words.
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
lemmatizer=WordNetLemmatizer()
input_str='been had done languages cities mice'
input_str=word_tokenize(input_str)
#for word in input_str:
#    print(lemmatizer.lemmatize(word))





# Part of speech tagging (POS)
#Part-of-speech tagging aims to assign parts of speech to each word of a given text
#(such as nouns, verbs, adjectives, and others) based on its definition and its context.
input_str='Parts of speech examples: an article, to write, interesting, easily, and, of'
from textblob import TextBlob
result = TextBlob(input_str)
#print(result.tags)





# Chunking (shallow parsing)
# Chunking is a natural language process that identifies constituent parts of sentences
# (nouns, verbs, adjectives, etc.) and links them to higher order units that have discrete grammatical meanings
#(noun groups or phrases, verb groups, etc.)
input_str='A black television and a white stove were bought for the new apartment of John.'
from textblob import TextBlob
result = TextBlob(input_str)
print(result.tags)

#The second step is chunking:
reg_exp = 'NP: {<DT>?<JJ>*<NN>}'
rp = nltk.RegexpParser(reg_exp)
result = rp.parse(result.tags)
print(result)

result.draw()





# Named entity recognition
# Named-entity recognition (NER)
# aims to find named entities in text and classify them into pre-defined categories
#(names of persons, locations, organizations, times, etc.).

ex = '''European authorities fined Google a record $5.1 billion on Wednesday
... for abusing its power in the mobile phone market and ordered
...the company to alter its practices''' 
def preprocess(sent):
    sent = nltk.word_tokenize(sent)
    sent = nltk.pos_tag(sent)
    return sent
sent = preprocess(ex)
print(sent)

#https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

pattern = 'NP: {<DT>?<JJ>*<NN>}'
#Chunking
cp = nltk.RegexpParser(pattern)
cs = cp.parse(sent)
print(cs)

from nltk.chunk import conlltags2tree, tree2conlltags
from pprint import pprint
iob_tagged = tree2conlltags(cs)
pprint(iob_tagged)

#ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
#print(ne_tree)

import spacy
from spacy import displacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
doc = nlp('''European authorities fined Google a record $5.1 billion
...on Wednesday for abusing its power in the mobile phone market and
...ordered the company to alter its practices''')
pprint([(X.text, X.label_) for X in doc.ents])
pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])









#Coreference resolution (anaphora resolution)
#Pronouns and other referring expressions should be connected to the right individuals.
#Coreference resolution finds the mentions in a text that refer to the same real-world entity.
#For example, in the sentence, “Andrew said he would buy a car”
#the pronoun “he” refers to the same person, namely to “Andrew”.

#Aqui falta buscar en web





#Collocation extraction
#Collocations are word combinations occurring together more often than would be expected by chance.
#Collocation examples are “break the rules,” “free time,” “draw a conclusion,” “keep in mind,” “get ready,” and so on.
#input=['he and Chazz duel with all keys on the line.']
#from ice import CollocationExtractor
#extractor = CollocationExtractor.with_collocation_pipeline('T1' , bing_key = 'Temp',pos_check = False)
#print(extractor.get_collocations_of_length(input, length = 3))





# Relationship extraction
#Relationship extraction allows obtaining structured information
#from unstructured sources such as raw text.
#Strictly stated, it is identifying relations (e.g., acquisition, spouse, employment)
#among named entities (e.g., people, organizations, locations)
#For example, from the sentence “Mark and Emily married yesterday,”
#we can extract the information that Mark is Emily’s husband.



# http://www.nltk.org/howto/relextract.html


#If you publish work that uses NLTK, please cite the NLTK book as follows:

#Bird, Steven, Edward Loper and Ewan Klein (2009), Natural Language Processing with Python. O’Reilly Media Inc.







