# from nltk.corpus import wordnet
from typing import List
import nltk
from nltk.wsd import lesk
from logger_utils import make_logger


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
