# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 17:08:19 2018

@author: Maxence Brochard
"""
#------------------------------------------------------------------------------
# Import packages and modules
#------------------------------------------------------------------------------
import re
import pickle
import sys
import os

#------------------------------------------------------------------------------
# SpellChecker
# Check and correct the spelling of words
# input:
#### words_list: words to correct
#### dictionary_wiki: computed from a large corpus (wikipedia). Used to compute the probability of a candidate to be chosen
#### dictionary_corpus_FR: all possible french words but slang words
#### dictionary_slang: common slang words (from wikipedia)
#### dictionary_urbandico: internet slang words (from urbandico)
#### dictionary_firstname: firstname
# output:
#### corrected words (from correct_text function)
#------------------------------------------------------------------------------

#TO ADD : dictionary nom propres

class SpellChecker:
    
    def __init__(self, words_list, WORDS, dictionary_corpus_FR, dictionary_slang, 
                 dictionary_urbandico, dictionary_firstname, dictionary_proper_nouns, dictionary_brands):
    
        self.words_list = words_list
        self.WORDS = WORDS
        self.dictionary_corpus_FR = dictionary_corpus_FR
        self.dictionary_slang = dictionary_slang
        self.dictionary_urbandico = dictionary_urbandico
        self.dictionary_firstname = dictionary_firstname
        self.dictionary_proper_nouns = dictionary_proper_nouns
        self.dictionary_brands = dictionary_brands
    
    # Get the subset of candidates words that are in the dictionary
    def known(self, words): 
        #return set(w for w in words if w in dictionary_corpus_FR, dictionary_slang)
        return set(w for w in words if any(w in i for i in (self.dictionary_corpus_FR, self.dictionary_slang, self.dictionary_urbandico, self.dictionary_firstname, self.dictionary_proper_nouns, self.dictionary_brands)))

    # Compute the frequency of a word within the large corpus
    def P(self, word):
        N=sum(self.WORDS.values())
        return self.WORDS[word] / N
    
    # Get all possible edits that are one edit away from word
    def edits1(self, word):
        letters    = 'abcdefghijklmnopqrstuvwxyzàâêîôûéèùëïüÿç'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    # Get all possible edits that are two edit away from word
    def edits2(self, word): 
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))

    # Get all possible candidates word 
    def candidates(self, word):
        candidates = (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])
        return candidates

    # Get the most probable spelling correction for word
    def correction(self, word):
        return max(self.candidates(word), key=self.P)

    # Return the case-function appropriate for text: upper, lower, title, or just str
    def case_of(self, text):
        return (str.upper if text.isupper() else
                str.lower if text.islower() else
                str.title if text.istitle() else
                str)

    # Spell-correct word in match, and preserve proper upper/lower/title case
    def correct_match(self, match):
        word = match.group()
        return self.case_of(word)(self.correction(word.lower()))
        
    # Correct all the words within a text, returning the corrected text
    def correct_text(self, text):
        text = re.sub('\x95',' ', text) #remove \x95 characters
        text = re.sub(' +',' ', text) #remove whitespaces
        return re.sub(r'(\w+)', self.correct_match, text)

#------------------------------------------------------------------------------
# load_obj
# Load a pickel object (CountVectorizer computed from wiki dump)
# input:
#### name: name of pickle object
# output:
#### pickle file
#------------------------------------------------------------------------------
def load_obj(name):
    with open(os.getcwd()+'/fonctions_outils/obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

#------------------------------------------------------------------------------
# load_file_into_dict
# load and convert a file in dict
# input:
#### path: path of the file
#### encoding: utf-8 or iso-8859-1 (depending on file)
# output:
#### dictionary
#------------------------------------------------------------------------------
def load_file_into_dict(path, encoding):
    dictionary = open(path, encoding=encoding).read().split('\n')
    dictionary = dict(zip(dictionary, range(len(dictionary)))) #quicker to look for a value into a dictionary O(1) vs O(n) than into a list
    return dictionary

# load all dictionaries
def load_dictionnaries():
    # Load counter object. Computed from a large corpus (wikipedia). Helps to determine which word candidates should be chosen
    WORDS = load_obj('wiki_words_occurences')
    # Load dictionary_corpus_FR. This allows to check whether the word candidate generated exists within common words
    dictionary_corpus_FR = load_file_into_dict(os.getcwd()+'/fonctions_outils/dictionary/dictionary_corpus_FR.txt', encoding = "utf-8")
    # Load dictionary_slang. This allows to check whether the word candidate generated exists within slang words
    dictionary_slang = load_file_into_dict(os.getcwd()+'/fonctions_outils/dictionary/dictionary_slang.txt', encoding = "utf-8")
    # Load dictionary_urbandico. This allows to check whether the word candidate generated exists within urbandico words
    dictionary_urbandico = load_file_into_dict(os.getcwd()+'/fonctions_outils/dictionary/dictionary_urbandico.txt', encoding = "iso-8859-1")
    # Load dictionary_firstname. This allows to check whether the word candidate generated exists within firstname
    dictionary_firstname = load_file_into_dict(os.getcwd()+'/fonctions_outils/dictionary/dictionary_firstname.txt', encoding = "iso-8859-1")
    # Load dictionary_proper_nouns. This allows to check whether the word candidate generated exists within proper nouns
    dictionary_proper_nouns = load_file_into_dict(os.getcwd()+'/fonctions_outils/dictionary/dictionary_proper_nouns.txt', encoding = "utf-8")
	# Load dictionary_brands. This allows to check whether the word candidate generated exists within brands
    dictionary_brands = load_file_into_dict(os.getcwd()+'/fonctions_outils/dictionary/dictionary_brands.txt', encoding = "utf-8")
    return WORDS, dictionary_corpus_FR, dictionary_slang, dictionary_urbandico, dictionary_firstname, dictionary_proper_nouns, dictionary_brands

def main(words_list):
    
    # load dictionaries
    WORDS, dictionary_corpus_FR, dictionary_slang, dictionary_urbandico, dictionary_firstname, dictionary_proper_nouns, dictionary_brands = load_dictionnaries()
    
    # create spell checker object
    spell_checker = SpellChecker(words_list, WORDS, dictionary_corpus_FR, dictionary_slang, 
                 dictionary_urbandico, dictionary_firstname, dictionary_proper_nouns, dictionary_brands)
    
    corrected_words = []    
    
    for word in words_list:
        corrected_words.append(spell_checker.correct_text(word))
    
    return corrected_words

if __name__ == "__main__":
    main(sys.argv[0:])
    