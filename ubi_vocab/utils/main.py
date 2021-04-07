import re
import pandas as pd
import plotly.express as px
from typing import List, Set, Dict, Tuple
from collections import defaultdict, namedtuple

from pre_process import SynWords
from article_class import ReplacedContext, Article
from data_io import get_clean_news, get_raw_vocab
from transformer import ST, calc_cos_sim
from logger_utils import make_logger

import spacy

# python -m spacy download en_core_web_sm
import en_core_web_sm

# from sense2vec
pd.set_option("display.max_colwidth", None)

# from https://github.com/explosion/spaCy/blob/master/spacy/glossary.py

SPACY_POS_MAP = {
    "ADJ": "adjective",
    "ADP": "adposition",
    "ADV": "adverb",
    "AUX": "auxiliary",
    "CONJ": "conjunction",
    "CCONJ": "coordinating conjunction",
    "DET": "determiner",
    "INTJ": "interjection",
    "NOUN": "noun",
    "PRON": "pronoun",
    "PROPN": "proper noun",
    "PUNCT": "punctuation",
    "SCONJ": "subordinating conjunction",
    "SYM": "symbol",
    "VERB": "verb",
    "JJ": "adjective",
    "JJR": "adjective",
    "JJS": "adjective",
    "ADJA": "adjective",
    "ADJD": "adjective",
    "ADVP": "adverb",
}


def eda_replacement(news_df: pd.DataFrame, word_syn_df: pd.DataFrame):
    """

    Args:
        news_df (pd.DataFrame): [news:str]
        word_syn_df (pd.DataFrame): [word: str, syn :str]

    Returns:
        [type]: [description]
    """

    # EDA - How many articles contain
    # For each synonym, count how many times they appear in a given article
    news_df["vocab"] = [set([])] * news_df.shape[0]  # number of matching unique vocab
    news_df["syn"] = [set([])] * news_df.shape[0]  # Number of matchin synonyms.
    news_df["num_syn"] = 0  # number of matching synonyms.
    gre_syn_set: Set[str] = set([x.lower() for x in word_syn_df["syn"]])

    vocab_appearances = pd.DataFrame()
    # Get the number of times vocabulary appear in the news dataset.
    for syn in gre_syn_set:
        # news_df.loc[news_df["article"].str.contains(syn), "num_vocab"] += 1
        # only match whole words.
        matches = news_df["article"].str.contains(
            f"\\b({syn})\\b", flags=re.IGNORECASE, regex=True
        )
        news_df.loc[matches, "num_syn"] += 1
        words_to_add = word_syn_df.query("syn == @syn").word.unique()
        #
        news_df.loc[matches, "vocab"] = news_df[matches]["vocab"].apply(
            lambda x: x.union(set(words_to_add))
        )

        news_df.loc[matches, "syn"] = news_df[matches]["syn"].apply(
            lambda x: x.union(set([syn]))
        )

        # Create a new article with replaced words.
        # news_df.loc[matches,
        #          "new_article"] = news_df[matches]\
        #                                 .apply(lambda x : x.article.replace(),
        #                                 axis = 1)

        # Calculate how many times a synonym appears in an article.
        vocab_appearances = vocab_appearances.append(
            pd.DataFrame({"syn": syn, "matches": matches.sum()}, index=[0])
        )

    vocab_appearances.sort_values("matches", ascending=False)

    # For a given vocab, how many news articles contain it's synonym.
    merged_df = vocab_appearances.merge(gre_syn, on="syn").sort_values(
        "matches", ascending=False
    )

    matches_df = (
        merged_df.groupby("word", as_index=False)
        .agg(matches=pd.NamedAgg("matches", "sum"))
        .sort_values("matches", ascending=False)
    )

    freq_plot = px.histogram(
        data_frame=matches_df,
        x="matches",
        title=f"Frequency of Vocab Synonyms that appear in {news_df.shape[0]} News Articles",
    )

    news_df["num_vocab"] = news_df["vocab"].apply(lambda x: len(x))

    return matches_df, merged_df, news_df, freq_plot
    # matches_df.query("matches > 0")
    # merged_df.query("word =='vacuous'")

    # Do a plotly distribution plot.


def get_cos_sim(
    text_a: List[str],
    text_b: List[str],
    st: ST = None,
    model_name: str = None,
    cache_dir: str = None,
) -> List[float]:
    """[summary]

    Args:
        text_a (List[str]): [description]
        text_b (List[str]): [description]
        st (ST) : sentence transformer object.
        model_name (str, optional): [description]. Defaults to None.
        cache_dir (str, optional): [description]. Defaults to None.

    Returns:
        (List[float]): List of normalized similarity scores.
    
    Example:
    >>> get_cos_sim(text_a = ["damaging effect", "positive effect"],
                    text_b = ["detrimental effect", "detrimental effect"],
                    )

    """
    log = make_logger(__name__)
    # sentence transformer:
    if st is None:
        st = ST(model_name=model_name)

    if len(text_a) != len(text_b):
        log.fatal(f"text_a {len(text_a)} must be the same len as text_b {len(text_b)}")
        assert len(text_a) == len(text_b)

    log.info(f"Starting to get embedding")
    embedded_a = st.encode(text_a)
    embedded_b = st.encode(text_b)

    log.info(f"Calculating cosine similarity scores")
    # TODO: This can be optimized if we want to do bulk scoring.
    sim_scores = [calc_cos_sim(a, embedded_b[i]) for i, a in enumerate(embedded_a)]
    return sim_scores


def _add_replaced_surrounding_sents(
    syn: str,
    word: str,
    sentences: List[str],
    sent_id: int,
    more_sent: int,
    total_sent: int,
    curr_orig_text: List[str] = [],
    curr_mod_text: List[str] = [],
) -> bool:
    """Given a list of sentences, a synonym and word to be replaced,
    and the number of surrounding sentences, add surrounding sentences
    to curr_orig_text and curr_mod_text.
    

    Args:
        syn (str): Synonym of a target vocabulary word found in an article.
        word (str): Target vocabulary word that will replace syn in the article.
        sentences (List[str]): List of all sentences in the article.
        sent_id (int): The index of the current sentence.
        more_sent (int): How many sentences before and after to include.
        total_sent (int): Total number of sentences. ie. len(sentences)
        curr_orig_text (List[str]): List of original text surrounding a target word.
        curr_mod_text (List[str]): List of modified text surrounding a target workd.
    
    Return:
        true if word was in found and replaced.
    """
    # Create whole word regex
    regex_syn = f"\\b({syn})\\b"
    sent = sentences[sent_id]
    if re.search(regex_syn, sent) is not None:
        # Find all sentences before and after the target sentence.
        first_sent = max(0, sent_id - more_sent)
        last_sent = min(sent_id + more_sent, total_sent)

        # If we only have one sentence, then just add one to get the
        # subsetting to work.
        if first_sent == last_sent:
            last_sent += 1
        # print(first_sent, last_sent, syn, sent)
        # print(" --------------------------------")

        orig_text = ". ".join(sentences[first_sent:last_sent])
        curr_orig_text.append(orig_text)

        # Replace highlighted text with new words
        curr_mod_text.append(re.sub(regex_syn, word, orig_text))
        return True
    return False


def _get_word_in_sent_pos(
    context: str, tar_word: str, spcy=None, pos_map: Dict[str, str] = SPACY_POS_MAP
) -> str:
    """
    # For a given raw sentence that contains a synonym,
    # Get the synonym's pos via syntactic parsing.

    # Returns a spacy pos. This is much more extensive than our pos.
    Args:
        context (str): [description]
        tar_word (str): [description]
        spcy ([type], optional): [description]. Defaults to None.

    Returns:
        str: [description]
    Examples:
    >>> context = "I played in a soccer game"
    >>> spcy = spacy.load("en_core_web_sm")
    >>> _get_word_in_sent_pos(context, tar_word = "game", spcy = spcy)    

    """
    if spcy is None:
        spcy = spacy.load("en_core_web_sm")

    for token in spcy(context):
        if token.text.lower() == tar_word.lower():
            if pos_map is None:
                return token.pos_
            else:
                return SPACY_POS_MAP[token.pos_]

    return None


# move to separate file.
def _get_sents_containing_word(
    all_text: str, word_syn_df: pd.DataFrame, num_sentences: int = 1, st=None, spcy=None
) -> List[ReplacedContext]:
    log = make_logger(__name__)
    if st is None:
        # st = sentence transformer object.
        st = ST(model_name=model_name)

    # word_syn_df contains a word and syn column.

    sentences: List[str] = all_text.split(".")

    # if we want more than one sentence surrounding the word,
    # grab num_sentences//2 before and after the word.
    more_sent = num_sentences // 2
    total_sent = len(sentences) - 1

    all_replaced_context = []

    # TODO: this can be much more efficient.
    (all_orig_text, all_mod_text, all_words, all_syn, all_context_pos, all_syn_pos) = (
        [],
        [],
        [],
        [],
        [],
        [],
    )

    # For each synonym, go through each sentence to see if it can be replaced.
    for tup in word_syn_df.itertuples():
        word: str = tup.word
        syn: str = tup.syn
        syn_pos: str = tup.syn_pos

        # A syn can occur in multiple places within an article.
        curr_orig_text, curr_mod_text, curr_pos = [], [], []
        for sent_id, sent in enumerate(sentences):
            # Fill curr_orig_text and curr_mod_text with neighboring sentences.
            is_replaced = _add_replaced_surrounding_sents(
                syn=syn,
                word=word,
                sentences=sentences,
                sent_id=sent_id,
                more_sent=more_sent,
                total_sent=total_sent,
                curr_orig_text=curr_orig_text,
                curr_mod_text=curr_mod_text,
            )

            # Extract the pos of the target word in the given sentence.
            if is_replaced:
                curr_pos.append(
                    _get_word_in_sent_pos(context=sent, tar_word=syn, spcy=spcy)
                )

        # There was a match, so add a replaced highlight.
        if len(curr_orig_text) > 0:

            # Append to master list.
            all_orig_text += curr_orig_text
            all_mod_text += curr_mod_text
            all_words += [word] * len(curr_orig_text)
            all_syn += [syn] * len(curr_orig_text)
            all_syn_pos += [syn_pos] * len(curr_orig_text)
            all_context_pos += curr_pos

            # TODO: consider removing.
            # all_replaced_context.append(ReplacedContext(vocab = word,
            #                                             syn = syn,
            #                                             orig_text = curr_orig_text,
            #                                             mod_text = curr_mod_text
            #                                             )

            #                                 )
    # Calc cosine similarity -  should be run in a more vectorized fashion.
    sim_scores = get_cos_sim(text_a=all_orig_text, text_b=all_mod_text, st=st)

    log.info(
        f"word: {len(all_words)}, syn : {len(all_syn)}, "
        f"orig_text : {len(all_orig_text)}, mod_text : {len(all_mod_text)}, "
        f"sim_score : {len(sim_scores)}"
    )
    # TODO: either store replaced context as a list of replacedcontext obj
    # or just as a dataframe.
    article_df = pd.DataFrame(
        dict(
            word=all_words,
            syn=all_syn,
            orig_text=all_orig_text,
            mod_text=all_mod_text,
            sim_score=sim_scores,
            syn_pos_context=all_context_pos,
            syn_pos=all_syn_pos,
        )
    )

    # for i, rv in enumerate(all_replaced_context):
    #     all_replaced_context[i].similarity_score = sim_scores[i]

    # return all_replaced_context
    return article_df


def get_replace_example(news_df: pd.DataFrame, word_syn_df: pd.DataFrame):

    seed = 28
    model_name = None
    tmp = news_df.sample(1, random_state=seed)
    news_df["original"] = news_df["article"]

    spcy = spacy.load("en_core_web_sm")
    #
    # bart = ST(model_name="facebook/bart-large-cnn")
    baseline = ST()
    orig_highlights: pd.DataFrame = _get_sents_containing_word(
        all_text=tmp["article"].array[0],
        word_syn_df=word_syn_df,
        num_sentences=1,
        st=baseline,
        spcy=spcy,
    )

    # pd.set_option('display.max_colwidth', None)
    # see how well scores perform here.
    bart_rv = px.histogram(orig_highlights["sim_score"])

    # \\b is necessary to match full words.
    # tmp["new_article"] = tmp.article.replace({f"\\b{k}\\b": v for k,v in syn_to_word_regex.items},
    #                                          regex = True)

    return orig_highlights


def main():
    # Read in raw vocab
    data_dir = "~/repos/ubiquitous_vocab/data/"
    small_news_file = os.path.join(data_dir, "news_small.csv")

    gre_df = get_raw_vocab(out_file=os.path.join(data_dir, "gre_vocab.csv"))

    # Read in news data from raw
    # news_df = get_clean_news(news_file= os.path.join(data_dir, "all-the-news-2-1.csv"))
    # seed = 28
    # small_df = news_df.sample(n =1000, random_state = seed)
    # small_df.to_csv(small_news_file, index = False)
    # Read in a small version of the news data.
    news_df = pd.read_csv(small_news_file)

    # Get synonyms of all gre df
    gre_syn_obj = SynWords(raw_data=gre_df)
    gre_syn = gre_syn_obj.get_synonyms()
    # syn_to_word = gre_syn_obj.get_syn_to_word()

    # Run eda module.
    # Look at 1000 news articles.
    # matched_df,\
    #  merged_df,\
    #  small_df,\
    #   freq_plot =  eda_replacement(news_df = news_df, word_syn_df = gre_syn)

    # Number of articles which will contain some vocab
    # px.histogram(small_df.num_vocab,
    #              title = "Number of Vocab words contained within an article")
    merged_df = pd.read_csv(os.path.join(data_dir, "tmp_small_merged_df.csv"))

    # Look at most popular synonyms.
    pop_syn = (
        merged_df.groupby("syn", as_index=False)
        .matches.sum()
        .sort_values(by="matches", ascending=False)
    )

    # Re-run the above with a narrower defined synset
    narrower_syn = set(pop_syn.query("matches < 300").syn.unique())

    matched_narr_df, merged_narr_df, small_narr_df, freq_narr_plot = eda_replacement(
        news_df=news_df, word_syn_df=gre_syn
    )

    px.histogram(
        small_narr_df.num_vocab,
        title="Number of Vocab words contained within an article",
    )

    small_narr_df.num_vocab.describe()

    # Perform the actual replacements. How are the replacements
    # doing? Do a sample and evaluate.

    #
    small_syn = gre_syn.query("syn.isin(@narrower_syn)")
    word_syn_df = small_syn
    # small_syn_to_word = {syn : word for syn, word in syn_to_word.items() if syn in narrower_syn}

    # Step through get_replace_example()
    replaced_sample = get_replace_example(news_df=news_df, word_syn_df=gre_syn)

