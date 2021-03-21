# Augment the raw vocab words by adding in synonyms,
#  maybe concordance lines
import nltk
# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd
from typing import List, Dict
from PyDictionary import PyDictionary as pyd
from collections import defaultdict

# There are many different ways to generate synonyms.
# Lets use 
# 1. nltk.wordnet systnet
# 2. py dict definition synonyms
# 3. maybe - word embeddings
# 4. 
class SynWords():
    """Maps words to supplamental information, such as synonyms, antynyms, and concordance lines.
    
    :example:
    >>> sys_obj = SynWords(words = ["perquisite", "unctuous"])
    >>> sys_obj.get_synonyms()
    >>> sys_obj.word_df
    """
    def __init__(self, words : List[str]):
        self.words = words
        # Use WordNet to map a word to it's synonym set.
        self.synset_map = { word : wn.synsets(word) for word in words}

        self.word_df = pd.DataFrame() # maps word to synonym, def, pos
        self.syn_to_word = defaultdict(list) # maps a synonym to word.

    def get_syn_to_word(self) -> Dict[str, List[str]]:
        # Requires get_synonyms to have been run.
        # Creates a mapping between synonym to a word.
        # This could be done in the synset for loop in get_synonyms
        # But putting it out here for simplicity.
        if self.word_df.shape[0] == 0:
            self.get_synonyms()

        for i in range(self.word_df.shape[0]):
            syn = self.word_df.iloc[i]["syn"]
            word = self.word_df.iloc[i]["word"]
            self.syn_to_word[syn].append(word)

        return self.syn_to_word

    def get_synonyms(self):
        # synset_map : Dict[str, List[nltk.corpus.reader.wordnet.Synset]]
        
        word_df = pd.DataFrame()
        for original, synsets in self.synset_map.items():
            # A synonym set contains multiple words called lemmas.
            # Build a synonym list by aggregating all lemmas.
            syn_list = []
            for synset in synsets:
                syn_list += [x for x in synset.lemma_names() if x != original]

            word_df = word_df.append(pd.DataFrame({"word" : original,
                                    "syn" : syn_list},
                                    index = [x for x in range(len(syn_list))]
                                        ))


        # Remove any duplicates
        word_df.drop_duplicates(inplace = True)
        word_df = word_df.query("word != syn")
        self.word_df = word_df
        return word_df
        
        # TODO: Need to rectify different ways of spelling the same word
        # eg. tendentious, tendencious
        # TODO: Pull additional words from pydict if not enough.
        # TODO: Pull in concordance
