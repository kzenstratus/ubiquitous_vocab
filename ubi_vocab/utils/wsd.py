# from nltk.corpus import wordnet
from typing import List
import nltk

import numpy as np
from nltk.wsd import lesk
from logger_utils import make_logger
from nltk.corpus import wordnet as wn
from transformer import ST
from constants import POS_MAP


def get_lesk(
    context_list: List[str], tar_word_list: List[str]
) -> List[nltk.corpus.reader.wordnet.Synset]:
    """Given a list of sentences, and a list of target words,
     use lesk to get the synset most closely aligned with the word in that context.

    Args:
        context_list (List[str]): [description]
        tar_word_list (List[str]): [description]

    Returns:
        List[synset]: [description]
    
    Example:
    >>> get_lesk(context_list = ["Steve Jobs founded Apple", "Apples are more tasty than oranges"],
                    tar_word_list = ["apple", "apple"],
                    )
    >>> # Here we see that apple isn't disambiguated through get_lesk()
    >>> get_lesk(context_list = ["I went to the bank to deposit my money", "I will bank my earnings"],
                    tar_word_list = ["bank", "bank"],
                    )
    >>> # Here bank does get correctly identified.
    """
    log = make_logger(__name__)

    if len(context_list) != len(tar_word_list):
        log.fatal(
            f"context_list {len(context_list)} must be the same len as tar_word_list {len(tar_word_list)}"
        )
        assert len(context_list) == len(tar_word_list)

    new_synsets = []
    # For each sentence and target word, get what lesk thinks is the wordnet synset
    # that the target word represents.
    for i, context in enumerate(context_list):
        sent = context.split()
        tar_word: str = tar_word_list[i]
        new_synsets.append(lesk(sent, tar_word))

    return new_synsets


# TODO: look at these methods for WSD. https://github.com/HSLCY/GlossBERT
def get_best_synset_bert(
    context_list: List[str], tar_word_list: List[str], st: ST, pos: List[str]
) -> List[nltk.corpus.reader.wordnet.Synset]:
    """Given a list of sentences, and a list of target words,
     use lesk to get the synset most closely aligned with the word in that context.

    Args:
        context_list (List[str]): [description]
        tar_word_list (List[str]): [description]
        st (ST) : sentence transformer object
        tar_post (List[str]): the pos of the word 

    Returns:
        List[synset]: [description]
    
    Example:
    >>> # Load in bert model
    >>> st = ST()
    >>> get_best_synset_bert(context_list = ["Steve Jobs founded Apple", "Apples are more tasty than oranges"],
                    tar_word_list = ["apple", "apple"],
                    st = st,
                    pos = ["noun", "noun"]
                    )
    >>> # Here we see that apple isn't disambiguated through bert either, however this is
    >>> # Because apple the company isn't defined in wordnet.

    >>> best_synsets, all_def = get_best_synset_bert(context_list = ["I went to the bank to deposit my money", "I will bank my earnings"],
                    tar_word_list = ["bank", "bank"],
                    st = st,
                    pos = ["noun", "verb"]
                    )
    >>> # Here lesk actually does better as the second bank doesn't get correctly matched.
    """
    log = make_logger(__name__)

    if len(context_list) != len(tar_word_list):
        log.fatal(
            f"context_list {len(context_list)} must be the same len as tar_word_list {len(tar_word_list)}"
        )
        assert len(context_list) == len(tar_word_list)

    all_def = pd.DataFrame()
    # For each sentence and target word, get what bert thinks is the wordnet synset
    # that the target word represents.
    for i, context in enumerate(context_list):
        tar_word: str = tar_word_list[i]
        tar_pos: str = pos[i]
        # Get all definitions for the target word.
        tmp_def = pd.DataFrame(dict(synsets=wn.synsets(tar_word)))
        # Only keep synsets which have the target pos.
        tmp_def["definition"] = tmp_def["synsets"].apply(
            lambda x: x.definition() if POS_MAP[x.pos()] == tar_pos else None
        )
        # Get word examples.
        tmp_def["example"] = tmp_def["synsets"].apply(
            lambda x: x.examples()[0]
            if POS_MAP[x.pos()] == tar_pos and len(x.examples()) > 0
            else None
        )
        tmp_def = tmp_def.query("definition.notnull()")
        # Create an index to do a groupby on. This is because get_cos_sim is best run once vectorized.
        tmp_def["idx"] = i
        tmp_def["context"] = context
        all_def = all_def.append(tmp_def)

    # Compare the word in context with all definitions of that word in wordnet.
    all_def["def_score"] = get_cos_sim(
        text_a=all_def["definition"], text_b=all_def["context"]
    )

    # Do a comparison with example sentences.
    is_example = all_def["example"].notnull()
    all_def.loc[is_example, "example_score"] = get_cos_sim(
        text_a=list(all_def[is_example]["example"]),
        text_b=list(all_def[is_example]["context"]),
        st=st,
    )

    # If example score isn't available, fill na with the definition score.
    all_def["score"] = all_def[["def_score", "example_score"]].mean(axis=1)

    best_synsets = list(
        all_def.sort_values("score", ascending=False)
        .groupby("idx")
        .head(1)
        .reset_index(drop=True)["synsets"]
    )

    return new_synsets, all_def
