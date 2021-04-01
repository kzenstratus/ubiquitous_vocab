# Augment the raw vocab words by adding in synonyms,
#  maybe concordance lines
import nltk

# nltk.download('wordnet')
from nltk.corpus import wordnet as wn
import pandas as pd
import pandera as pa
from typing import List, Dict
from PyDictionary import PyDictionary as pyd
from collections import defaultdict


POS_MAP = dict(s="adjective", n="noun", r="adverb", a="adjective", v="verb")

# There are many different ways to generate synonyms.
# Lets use
# 1. nltk.wordnet systnet
# 2. py dict definition synonyms
# 3. maybe - word embeddings
# 4.
class SynWords:
    """Maps words to supplamental information, such as synonyms, antynyms, and concordance lines.
    
    :example:
    >>> sys_obj = SynWords(words = ["perquisite", "unctuous"])
    >>> sys_obj.get_synonyms()
    >>> sys_obj.word_df
    """

    def __init__(self, raw_data: pd.DataFrame):

        # Limit word usage to their specific pos.
        schema = pa.DataFrameSchema(
            {"word": pa.Column(pa.String), "pos": pa.Column(pa.String)}
        )
        schema.validate(raw_data)
        self.raw_data = raw_data
        # self.words = words
        # self.pos = pos
        # Use WordNet to map a word to it's synonym set.
        self.synset_map = {word: wn.synsets(word) for word in words}

        self.word_df = pd.DataFrame()  # maps word to synonym, def, pos
        self.syn_to_word = defaultdict(list)  # maps a synonym to word.

    def get_synset_map(raw_data: pd.DataFrame = None):
        if raw_data is None:
            raw_data = self.raw_data
        # Generate a mapping from word to synset.
        # Limit synset to only synsets which have the same pos.
        # raw_data.reset_index(drop = True, inplace = True)
        synset_list = []
        for i, raw_word in enumerate(raw_data["word"]):
            synset_list.append(
                pd.DataFrame(
                    dict(
                        word=raw_word,
                        pos=raw_data.iloc[i]["pos"],
                        synset=wn.synsets(raw_word),
                    )
                )
            )
        synset_df = pd.concat(synset_list)
        # Only keep synonyms that have matching parts of speech.
        synset_df[
            synset_df.apply(lambda row: row.pos == POS_MAP[row.synset.pos()], axis=1)
        ]
        return synset_df

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

    def get_synonyms(self, synset_map: pd.DataFrame = None):
        if synset_map is None:
            synset_map = self.synset_map
        # Expand the set of related synonyms by looking at lemma of synsets.
        # synset_map : Dict[str, List[nltk.corpus.reader.wordnet.Synset]]
        additional_syn = []
        for i, synset in enumerate(synset_map["synset"]):
            original = synset_map.iloc[i]["word"]
            curr_pos = synset_map.iloc[i]["pos"]
            # Only add additional synonyms if not the original and has matching pos.
            syn_list = [
                x.synset()
                for x in synset.lemmas()
                if x.name != original and POS_MAP[x.synset().pos()] == curr_pos
            ]
            additional_syn.append(
                pd.DataFrame(dict(word=original, pos=curr_pos, synset=syn_list))
            )

        # Append additional synonyms to the original synset map.
        synset_map.append(pd.concat(additional_syn))

        # Extract the name of the synset map.
        synset_map["syn"] = synset_map["synset"].apply(
            lambda x: re.sub("^([a-z]*)\\..*", "\\1", x.name())
        )
        # Extract the syn pos
        synset_map["syn_pos"] = synset_map["synset"].apply(lambda x: x.pos())

        # Remove any duplicates
        # NOTE: we may not want to remove duplicates if we want to utilize
        # synset.

        synset_map.drop_duplicates(subset=["word", "syn"], inplace=True)
        synset_map = synset_map.query("word != syn")
        self.synset_map_full = synset_map
        return synset_map

        # TODO: Need to rectify different ways of spelling the same word
        # eg. tendentious, tendencious
        # TODO: Pull additional words from pydict if not enough.
        # TODO: Pull in concordance
