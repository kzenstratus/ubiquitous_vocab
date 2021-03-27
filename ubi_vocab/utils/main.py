import re
import pandas as pd
import plotly.express as px
from typing import List, Set, Dict, Tuple
from collections import defaultdict, namedtuple

from pre_process import SynWords
from article_class import ReplacedContext, Article
from data_io import get_clean_news, get_raw_vocab


def eda_replacement(news_df : pd.DataFrame,
                    word_syn_df : pd.DataFrame):
    """

    Args:
        news_df (pd.DataFrame): [news:str]
        word_syn_df (pd.DataFrame): [word: str, syn :str]

    Returns:
        [type]: [description]
    """
    
    # EDA - How many articles contain 
    # For each synonym, count how many times they appear in a given article
    news_df["vocab"] = [set([])] * news_df.shape[0] # number of matching unique vocab
    news_df["syn"] = [set([])] * news_df.shape[0] # Number of matchin synonyms.
    news_df["num_syn"] =  0# number of matching synonyms.
    gre_syn_set : Set[str] = set([x.lower() for x in word_syn_df["syn"]])

    vocab_appearances = pd.DataFrame()
    # Get the number of times vocabulary appear in the news dataset.
    for syn in gre_syn_set:
        # news_df.loc[news_df["article"].str.contains(syn), "num_vocab"] += 1
        # only match whole words.
        matches = news_df["article"]\
                        .str.contains(f"\\b({syn})\\b",
                           flags=re.IGNORECASE,
                           regex = True )
        news_df.loc[matches, "num_syn"] += 1
        words_to_add = word_syn_df.query("syn == @syn").word.unique()
        # 
        news_df.loc[matches, "vocab"] = news_df[matches]["vocab"]\
                                                .apply(lambda x : x.union(set(words_to_add)))

        news_df.loc[matches, "syn"] = news_df[matches]["syn"]\
                                                .apply(lambda x : x.union(set([syn])))

        # Create a new article with replaced words.
        # news_df.loc[matches,
        #          "new_article"] = news_df[matches]\
        #                                 .apply(lambda x : x.article.replace(),
        #                                 axis = 1)

        # Calculate how many times a synonym appears in an article.
        vocab_appearances = vocab_appearances.append(
                                pd.DataFrame({"syn" : syn,
                                            "matches" : matches.sum()},
                                            index = [0])
                                            
                                )

    vocab_appearances.sort_values("matches", ascending = False)
    
    # For a given vocab, how many news articles contain it's synonym.
    merged_df = vocab_appearances\
                    .merge(gre_syn, on = "syn")\
                    .sort_values("matches", ascending = False)
    
    matches_df = merged_df\
                    .groupby("word", as_index= False)\
                    .agg(matches = pd.NamedAgg("matches", "sum"))\
                    .sort_values("matches", ascending = False)

    freq_plot = px.histogram(data_frame = matches_df,
                 x = "matches",
                 title = f"Frequency of Vocab Synonyms that appear in {news_df.shape[0]} News Articles"
                 )
    
    news_df["num_vocab"] = news_df["vocab"].apply(lambda x : len(x))

    return matches_df, merged_df, news_df, freq_plot
    # matches_df.query("matches > 0")
    # merged_df.query("word =='vacuous'")    

    # Do a plotly distribution plot.
    

def _add_replaced_surrounding_sents(syn : str,
                                    word : str,
                                    sentences : List[str],
                                    sent_id : int,
                                    more_sent: int,
                                    total_sent : int,
                                    all_orig_text : List[str] = [],
                                    all_mod_text : List[str] = []) -> None:
    """Given a list of sentences, a synonym and word to be replaced,
    and the number of surrounding sentences, add surrounding sentences
    to all_orig_text and all_mod_text.
    

    Args:
        syn (str): Synonym of a target vocabulary word found in an article.
        word (str): Target vocabulary word that will replace syn in the article.
        sentences (List[str]): List of all sentences in the article.
        sent_id (int): The index of the current sentence.
        more_sent (int): How many sentences before and after to include.
        total_sent (int): Total number of sentences. ie. len(sentences)
        all_orig_text (List[str]): List of original text surrounding a target word.
        all_mod_text (List[str]): List of modified text surrounding a target workd.
    """
    # Create whole word regex
    regex_syn = f"\\b({syn})\\b"

    if re.search(regex_syn, sent) is not None:
        # Find all sentences before and after the target sentence.
        first_sent = max(0, sent_id - more_sent)
        last_sent = min(sent_id + more_sent, total_sent)
        
        # If we only have one sentence, then just add one to get the 
        # subsetting to work.
        if first_sent == last_sent:
            last_sent += 1
        print(first_sent, last_sent, syn, sent)
        print(" --------------------------------")
        
        orig_text = ". ".join(sentences[first_sent : last_sent])
        all_orig_text.append(orig_text)
        
        # Replace highlighted text with new words
        all_mod_text.append(re.sub(regex_syn, word , orig_text))


# move to separate file.
def _get_sents_containing_word(all_text : str,
                                word_syn_df : pd.DataFrame,
                                num_sentences : int = 1
                                ) -> List[ReplacedContext]:
    # word_syn_df contains a word and syn column.
    
    sentences : List[str] = all_text.split(".")

    # if we want more than one sentence surrounding the word,
    # grab num_sentences//2 before and after the word.
    more_sent = num_sentences // 2
    total_sent = len(sentences) - 1
    
    all_replaced_context = []
    # For each synonym, go through each sentence to see if it can be replaced.   
    for tup in word_syn_df.itertuples():
        word : str = tup.word
        syn : str = tup.syn

        all_orig_text = []
        all_mod_text = []
        for sent_id, sent in enumerate(sentences):
            # Fill all_orig_text and all_mod_text with neighboring sentences.
            _add_replaced_surrounding_sents(syn = syn,
                                            word = word,
                                            sentences = sentences,
                                            sent_id = sent_id,
                                            more_sent = more_sent,
                                            total_sent = total_sent,
                                            all_orig_text=all_orig_text,
                                            all_mod_text=all_mod_text)
            

        # There was a match, so add a replaced highlight.
        if len(all_orig_text) > 0:
            all_replaced_context.append(ReplacedContext(vocab = word,
                                                        syn = syn,
                                                        orig_text = all_orig_text,
                                                        mod_text = all_mod_text)
                                            )

    return all_replaced_context

def get_replace_example(news_df : pd.DataFrame,
                        syn_to_word : Dict[str, List[str]]):

    # For now, lets limit the original word to the first one.
    # syn_to_word_regex : Dict[str, str] = {syn: word[0] for syn, word in syn_to_word.items()}
    # word_to_syn_regex : Dict[str, str] = {word[0] : syn for syn, word in syn_to_word.items()}
    seed = 28

    tmp = news_df.sample(1, random_state = seed)
    news_df["original"] = news_df["article"]
    
    # 
    orig_highlights : List[ReplacedContext] = _get_sents_containing_word(tmp["article"].array[0],
                                                word_syn_df = syn_to_word)
    
    # tmp["orig_highlights"] = orig_highlights

    # \\b is necessary to match full words.    
    tmp["new_article"] = tmp.article.replace({f"\\b{k}\\b": v for k,v in syn_to_word_regex.items},
                                             regex = True)

    # mod_highlights = _get_sents_containing_word(tmp["new_article"].array[0],
    #                             word_map = word_to_syn_regex)


    # news_df.sample(1, random_state = seed)\
    #         .article\
    #         .replace(syn_to_word, regex = True, inplace = True)

    # Highlight the replaced sentence.


def main():
    # Read in raw vocab
    data_dir = "~/repos/ubiquitous_vocab/data/"
    small_news_file = os.path.join(data_dir, "news_small.csv")

    gre_df = get_raw_vocab(out_file = os.path.join(data_dir, "gre_vocab.csv"))

    # Read in news data from raw
    # news_df = get_clean_news(news_file= os.path.join(data_dir, "all-the-news-2-1.csv"))
    # seed = 28
    # small_df = news_df.sample(n =1000, random_state = seed)
    # small_df.to_csv(small_news_file, index = False)
    # Read in a small version of the news data.
    news_df = pd.read_csv(small_news_file)

    # Get synonyms of all gre df
    gre_syn_obj = SynWords(words = gre_df["word"])
    gre_syn = gre_syn_obj.get_synonyms()
    syn_to_word = gre_syn_obj.get_syn_to_word()

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
    pop_syn = merged_df.groupby("syn", as_index = False)\
                        .matches.sum()\
                        .sort_values(by = "matches", ascending = False)

    # Re-run the above with a narrower defined synset
    narrower_syn = set(pop_syn.query("matches < 300").syn.unique())
    
    matched_narr_df,\
     merged_narr_df,\
     small_narr_df,\
      freq_narr_plot =  eda_replacement(news_df = news_df,
       word_syn_df = gre_syn.query("syn.isin(@narrower_syn)"))
    
    px.histogram(small_narr_df.num_vocab, title = "Number of Vocab words contained within an article")
    
    small_narr_df.num_vocab.describe()

    # Perform the actual replacements. How are the replacements
    # doing? Do a sample and evaluate.

    # 
    small_syn = gre_syn.query("syn.isin(@narrower_syn)")
    # small_syn_to_word = {syn : word for syn, word in syn_to_word.items() if syn in narrower_syn}


    def get_cos_sim(text_a : List[str],
                    text_b : List[str],
                    model_name : str = None,
                    cache_dir : str = None) -> List[float]:
        """[summary]

        Args:
            text_a (List[str]): [description]
            text_b (List[str]): [description]
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
        st = ST(model_name = model_name)

        if len(text_a) != len(text_b):
            log.fatal(f"text_a {len(text_a)} must be the same len as text_b {len(text_b)}")
            assert(len(text_a) == len(text_b))

        log.info(f"Starting to get embedding")        
        embedded_a = st.encode(text_a)
        embedded_b = st.encode(text_b)

        log.info(f"Calculating cosine similarity scores")
        # TODO: This can be optimized if we want to do bulk scoring.
        sim_scores = [calc_cos_sim(a, embedded_b[i]) for i, a in enumerate(embedded_a)]
        return sim_scores