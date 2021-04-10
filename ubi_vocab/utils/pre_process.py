# Augment the raw vocab words by adding in synonyms,
#  maybe concordance lines
import nltk, re

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
        self.synset_map = self.get_synset_map(raw_data=raw_data)
        self.synset_map = self.get_synonyms(synset_map=self.synset_map)
        # self.word_df = pd.DataFrame()  # maps word to synonym, def, pos
        self.syn_to_word = defaultdict(list)  # maps a synonym to word.

    def get_synset_map(self, raw_data: pd.DataFrame = None):
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
                        word_def=raw_data.iloc[i]["definition"],
                    )
                )
            )
        synset_df = pd.concat(synset_list)

        # Only keep synonyms that have matching parts of speech.
        synset_df = synset_df[
            synset_df.apply(lambda row: row.pos == POS_MAP[row.synset.pos()], axis=1)
        ]

        return synset_df

    # def get_syn_to_word(self) -> Dict[str, List[str]]:
    #     # Requires get_synonyms to have been run.
    #     # Creates a mapping between synonym to a word.
    #     # This could be done in the synset for loop in get_synonyms
    #     # But putting it out here for simplicity.
    #     if self.word_df.shape[0] == 0:
    #         self.get_synonyms()

    #     for i in range(self.word_df.shape[0]):
    #         syn = self.word_df.iloc[i]["syn"]
    #         word = self.word_df.iloc[i]["word"]
    #         self.syn_to_word[syn].append(word)

    #     return self.syn_to_word

    def get_synonyms(self, synset_map: pd.DataFrame = None):
        if synset_map is None:
            synset_map = self.synset_map

        # Get the all synonyms by looking at words that are related to the definition
        # of a specific synset. This adds the syn column.
        # synset_map : Dict[str, List[nltk.corpus.reader.wordnet.Synset]]
        additional_syn = []

        # TODO :there is probably a pandas native way to do this.
        for tup in synset_map.itertuples():
            original: str = tup.word
            curr_pos: str = tup.pos
            synset: str = tup.synset
            # Only add additional synonyms if not the original and has matching pos.
            syn_list = [
                x.name()
                for x in synset.lemmas()
                if x.name != original and POS_MAP[x.synset().pos()] == curr_pos
            ]

            additional_syn.append(
                pd.DataFrame(
                    dict(
                        word=original,
                        pos=curr_pos,
                        synset=synset,
                        syn=syn_list,
                        word_def=tup.word_def,
                    )
                )
            )

        # Append additional synonyms to the original synset map.
        syn_df = pd.concat(additional_syn)

        # Extract the definition of the synset map.
        syn_df["syn_def"] = syn_df["synset"].apply(lambda x: x.definition())

        # Extract the syn pos
        syn_df["syn_pos"] = syn_df["synset"].apply(lambda x: POS_MAP[x.pos()])

        # Remove any duplicates
        # NOTE: we may not want to remove duplicates if we want to utilize
        # synset.

        syn_df = syn_df.query("word != syn")

        syn_df.drop_duplicates(subset=["word", "syn", "pos"], inplace=True)

        self.syn_df = syn_df
        return syn_df

        # TODO: Need to rectify different ways of spelling the same word
        # eg. tendentious, tendencious
        # TODO: Pull additional words from pydict if not enough.
        # TODO: Pull in concordance
