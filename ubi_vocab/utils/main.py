import re, os
import pandas as pd
import plotly.express as px
from typing import List, Set, Dict, Tuple
from collections import defaultdict, namedtuple

from pre_process import SynWords
from article_class import ReplacedContext, Article
from data_io import get_clean_news, get_raw_vocab
from transformer import ST, calc_cos_sim, get_cos_sim
from logger_utils import make_logger
from constants import SPACY_POS_MAP
from metrics import Metrics

from wsd import get_lesk, get_best_synset_bert
import spacy

# python -m spacy download en_core_web_sm
import en_core_web_sm

# from sense2vec
pd.set_option("display.max_colwidth", None)


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
    >>> spcy = en_core_web_sm.load()
    >>> _get_word_in_sent_pos(context, tar_word = "game", spcy = spcy)    

    """
    if spcy is None:
        spcy = en_core_web_sm.load()

    for token in spcy(context):
        if token.text.lower() == tar_word.lower():
            if pos_map is None:
                return token.pos_
            else:
                return SPACY_POS_MAP[token.pos_]

    return None


# move to separate file.
def _get_sents_containing_word(
    all_text: str,
    word_syn_df: pd.DataFrame,
    num_sentences: int = 1,
    st: ST = None,
    spcy=None,
) -> pd.DataFrame:
    """For a given news article/text, get (num_sentences) surrounding
    sentences around all synonyms listed in word_syn_df. Create a modified sentence
    which contains the original sentence(s) with the synonym of a target vocabulary word
    replaced with that vocabulary word.

    - Use Spacy to get POS values for the target word in the sentence in all_text.
    - Use BERT from the sentence transformer module (st) to calculate similarity score
    between the original sentence(s) and the modified sentence.

    Args:
        all_text (str): a news article
        word_syn_df (pd.DataFrame): 
        [word : str, syn : str, syn_pos: str]
        A dataframe mapping target word to synonym.
        num_sentences (int, optional): How many sentences before/after the sentence
        containing a target word to include when highlighting a replacement. Defaults to 1.
        st (ST, optional): sentence transformer object defined in transformer. Defaults to None.
        spcy ([type], optional): Spacy model used to get POS. Defaults to None.

    Returns:
        pd.DataFrame: 
            word : str = target vocabulary word,
            syn : str = synonym of vocabulary word found in original text,
            orig_text :str = num_sentences surrounding the syn in all_text,
            mod_text : str = orig_text with the syn replaced by word,
            sim_score : float = cosine similarity score comparing vectorized orig_text and mod_text,
            syn_pos_context : str = pos of the syn in orig_text,
            syn_pos : str = pos of the intended syn. If this doesn't match syn_pos_context, this would
            be a bad replacement.,

    """
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

    return article_df


def get_replace_example(
    news_df: pd.DataFrame,
    word_syn_df: pd.DataFrame,
    seed: int = None,
    sample_size: int = 1,
    bert_model_name: str = None,
    spacy_model=en_core_web_sm,
    num_sentences: int = 1,
    run_lesk_wsd: bool = True,
    run_bert_wsd: bool = True,
) -> pd.DataFrame:
    """

    Args:
        news_df (pd.DataFrame): 
        word_syn_df (pd.DataFrame): [word : str, syn : str, syn_pos: str]
        seed (int, optional): [description]. Defaults to None.
        sample_size (int, optional): [description]. Defaults to 1.
        bert_model_name (str, optional): [description]. Defaults to None.
        spacy_model (str, optional): [description]. Defaults to "en_core_web_sm".

    Returns:
        List[dict]: [{"article" : str, "highlights_df" : pd.DataFrame}]

    Example:
    >>> news_df = pd.DataFrame(dict(article = [(f"This is an example news article which " 
                                        f"shows the damaging effects of being " 
                                        f"unable to learn vocabulary."),
                                        (f"Learning vocabulary by rote memorization has a passing "
                                        f"effect on long term vocabulary understanding.")
                                        ]
                            ))
    # Get the word to synonym df.
    >>> gre_syn_obj = SynWords(raw_data=gre_df)
    >>> gre_syn = gre_syn_obj.get_synonyms()

    >>> ubi_vocab : List[dict] = get_replace_example(news_df = news_df,
                            word_syn_df = gre_syn)
    """
    log = make_logger(__name__)

    if seed is not None:
        tmp = news_df.sample(sample_size, random_state=seed)
        log.info(f"sampling down news_df to {tmp.shape} using seed {seed}")
    else:
        tmp = news_df

    # Load Spacy and BERT models.
    log.info(f"Loading in spacy and sentence transformer models.")

    # For some reason finding the model by string doesn't work well on binder.

    spcy = spacy_model.load()

    st = ST(model_name=bert_model_name)

    # For each news article,
    # Highlight sentences within an article that have replaced words.
    rv: List[dict] = []
    log.info(f"Getting replaced sentences for each news article.")
    for row in tmp.itertuples():
        article: str = row.article

        # Get one article's worth of replaced sentences.
        highlights: pd.DataFrame = _get_sents_containing_word(
            all_text=article,
            word_syn_df=word_syn_df,
            num_sentences=num_sentences,
            st=st,
            spcy=spcy,
        )

        # Add the synset to results.
        highlights = highlights.merge(
            word_syn_df[["word", "syn", "synset"]], on=["word", "syn"], how="left"
        )

        # Replacements should only occur when pos is the same.
        good_replace_query = ["syn_pos_context == syn_pos"]

        if run_lesk_wsd:
            # Word sense disambiguation comparing sentence used in a sentence with
            # the intended usage of a word (utilizing lesk/wordnet).
            log.info(f"Adding LESK results")
            highlights["lesk"] = get_lesk(
                context_list=highlights["mod_text"], tar_word_list=highlights["word"]
            )

            # Only make replacements when the synset is the same as the lesk results.
            good_replace_query.append("synset == lesk")

        if run_bert_wsd:
            # Compare the word definition to the sentence containing the original sentence.
            # Return the best synset based on cosine similarity to the target word definition
            # and example sentences.

            best_synsets, _ = get_best_synset_bert(
                context_list=highlights["mod_text"],
                tar_word_list=highlights["word"],
                st=st,
                pos=highlights["syn_pos"],
            )
            highlights["bert_wsd"] = best_synsets
            # The best results don't use synset and bert_wsd.
            good_replace_query.append("synset == bert_wsd")
            # TODO: want to compare to the averge sentences of all sentence use cases SEMCOR.
            # Instead let's use the rank the similarity with each synset dictionary definition.
            # And choose the highest score.

        to_replace: pd.DataFrame = highlights.copy()
        for q in good_replace_query:
            to_replace = to_replace.query(q)
        to_replace = to_replace[["word", "syn"]]

        log.info(f"Replacing the entire article {to_replace.shape}")
        # \\b is necessary to match full words.
        # too lazy to use re.sub, just use pandas for replacement.
        new_article: str = pd.Series(article).replace(
            {f"\\b{t.syn}\\b": t.word for t in to_replace.itertuples()}, regex=True
        )

        rv.append(
            dict(article=article, highlights_df=highlights, new_article=new_article)
        )

    # pd.set_option('display.max_colwidth', None)
    # see how well scores perform here.
    # bart_rv = px.histogram(orig_highlights["sim_score"])

    return rv


def eval_different_filters(data_dir: str) -> pd.DataFrame:
    """[summary]

    Args:
        data_dir (str): "../../data/"

    Returns:
        pd.DataFrame: [description]
    """
    master_file = os.path.join(data_dir, "master_labeled.csv")
    master_df = pd.read_csv(master_file)

    # Get the Synonym object for WSD.
    gre_df = get_raw_vocab(out_file=os.path.join(data_dir, "gre_vocab.csv"))
    gre_syn_obj = SynWords(raw_data=gre_df)
    gre_syn = gre_syn_obj.get_synonyms()

    # Add in original synset
    master_df = master_df.merge(
        gre_syn[["word", "syn", "synset"]], on=["word", "syn"], how="left"
    )

    # Add pos matching.
    master_df["pos_score"] = master_df["syn_pos"] == master_df["syn_pos_context"]
    master_df["pos_score"] = master_df["pos_score"].astype(int)

    # Get the replaced sentences to evaluate word sense disambiguation.

    # Get recommendation based on lesk results.
    master_df["lesk_score"] = master_df["lesk"] == master_df["synset"].astype(str)
    master_df["lesk_score"] = master_df["lesk_score"].astype(int)

    # Get recommendation based on BERT WSD.
    master_df["bert_wsd_score"] = master_df["bert_wsd"] == master_df["synset"].astype(
        str
    )
    master_df["bert_wsd_score"] = master_df["bert_wsd_score"].astype(int)

    # Create an aggregate score for POS + LESK
    master_df["pos_lesk_score"] = 0
    master_df.loc[
        (master_df["lesk_score"] == master_df["pos_score"])
        & (master_df["lesk_score"] == 1),
        "pos_lesk_score",
    ] = 1
    # POS + BERT WSD
    master_df["pos_bert_score"] = 0
    master_df.loc[
        (master_df["bert_wsd_score"] == master_df["pos_score"])
        & (master_df["bert_wsd_score"] == 1),
        "pos_bert_score",
    ] = 1

    # POS + BERT WSD + LESK
    master_df["pos_bert_lesk_score"] = 0
    master_df.loc[
        (master_df["pos_lesk_score"] == 1) & (master_df["pos_bert_score"] == 1),
        "pos_bert_lesk_score",
    ] = 1

    master_df["base"] = 1
    # Get the TP/FP/Precision/F1 of each.
    metrics = Metrics(labels=master_df["label"])
    metrics_df = pd.DataFrame()

    for model in [
        "base",
        "pos_score",
        "lesk_score",
        "bert_wsd_score",
        "pos_lesk_score",
        "pos_bert_score",
        "pos_bert_lesk_score",
    ]:
        metrics.get_pr_f1(scores=master_df[model])
        metrics_df = metrics_df.append(
            pd.DataFrame(
                dict(
                    model=[model],
                    precision=[metrics.precision],
                    recall=[metrics.recall],
                    f1=[metrics.f1],
                    tp=[metrics.tp],
                    fp=[metrics.fp],
                    fn=[metrics.fn],
                    tn=[metrics.tn],
                )
            )
        )

    return master_df, metrics_df


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
    replaced_sample_large = get_replace_example(
        news_df=news_df, word_syn_df=gre_syn, seed=28, sample_size=1
    )[0]["highlights_df"]

    # labeled_df = pd.read_csv(os.path.join(data_dir, "master_labeled.csv"))
    # labeled_df = labeled_df.merge(replaced_sample_large[["word", "syn", "orig_text", "lesk", "bert_wsd"]],
    #                 on = ["word", "syn", "orig_text"],
    #                 how = "left")
    # replaced_sample_large.to_csv(os.path.join(data_dir, "master_labeled.csv"), index = False)

    replaced_sample = get_replace_example(
        news_df=news_df, word_syn_df=small_syn, seed=28, sample_size=1
    )

    # replaced_sample = orig_highlights
    # replaced_sample.query("syn_pos_context == syn_pos").to_csv(
    #     os.path.join(data_dir, "labeled_pos_matched_2.csv"), index=False
    # )

    # Evaluate LESK
    # Reduce sampled df based on matching pos.
    lesk_df = replaced_sample_large.query("syn_pos_context == syn_pos").reset_index(
        drop=False
    )
    lesk_df["lesk"] = get_lesk(
        context_list=lesk_df["mod_text"], tar_word_list=lesk_df["word"]
    )
    lesk_df = lesk_df.merge(
        gre_syn[["word", "syn", "synset"]], on=["word", "syn"], how="left"
    )

    lesk_df = lesk_df.query("synset == lesk").reset_index(drop=True)
    lesk_df.to_csv(os.path.join(data_dir, "labeled_lesk.csv"), index=False)
    # 4 TP , 8 FP, 33% precision

    # Let's do cosine similarity of the modified text and the original definition.

    # reduced to 12 from 23.

    # Get the synset of the word/syn
