# This script contains the user facing interface to take a body of text, and get the replaced results.
import os
import pandas as pd
import plotly.express as px
from typing import List, Set, Dict, Tuple

from logger_utils import make_logger
from main import get_replace_example

from data_io import get_raw_vocab
from pre_process import SynWords


# python -m spacy download en_core_web_sm


class UbiVocab:
    """Main interface for replacing words in a body of text with 
    gre vocabulary words.
    
    Example:
    >>> data_dir = "~/repos/ubiquitous_vocab/data/"
    >>> ubi_vocab = UbiVocab(data_dir = data_dir)
    >>> article = (f"This is an article about the damaging effects of not learning vocabulary. "
                    f" This program seeks to create a routine exercise for learning vocabulary over time.")
    >>> ubi_vocab.process_article(article = article, num_sentences = 1)
    >>> # Look at the new "article" with replaced words.
    >>> ubi_vocab.new_article
    >>> # Look under the hood at how the program came to this conclusion.
    >>> ubi_vocab.highlights_df

    """

    def __init__(self, data_dir: str) -> None:

        self.gre_df = get_raw_vocab(out_file=os.path.join(data_dir, "gre_vocab.csv"))
        self.gre_syn_obj = SynWords(raw_data=self.gre_df)
        self.gre_syn = self.gre_syn_obj.get_synonyms()

    def process_article(
        self,
        article: str,
        run_lesk_wsd: bool = True,
        run_bert_wsd: bool = False,
        num_sentences: int = 1,
    ) -> None:
        # Format as a dataframe.
        news_df = pd.DataFrame(dict(article=[article]))
        rv = get_replace_example(
            news_df=news_df,
            word_syn_df=self.gre_syn,
            run_bert_wsd=run_bert_wsd,
            run_lesk_wsd=run_lesk_wsd,
        )[0]
        self.original_article = rv["article"]
        self.new_article = rv["new_article"]
        self.highlights_df: pd.DataFrame = rv["highlights_df"]

