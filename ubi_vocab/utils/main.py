import re
import pandas as pd

from pre_process import SynWords
from data_io import get_news, get_raw_vocab


def main():
    # Read in raw vocab
    data_dir = "~/repos/ubiquitous_vocab/data/"
    gre_df = get_raw_vocab(out_file = os.path.join(data_dir, "gre_vocab.csv"))

    # Read in news data
    news_df = get_clean_news(news_file= os.path.join(data_dir, "all-the-news-2-1.csv"))

    # Get synonyms of all gre df
    gre_syn_obj = SynWords(words = gre_df["word"])
    gre_syn = gre_syn_obj.get_synonyms()

    # Get all unique lowercase synonyms.
    gre_syn_set = set([x.lower() for x in gre_syn["syn"]])

    # EDA - How many articles contain 
    # For each synonym, count how many times they appear in a given article
    news_df["num_vocab"] = 0
    small_df = news_df[:10]
    small_df["num_vocab"] = 0
    

    vocab_appearances = pd.DataFrame()
    for syn in gre_syn_set:
        # news_df.loc[news_df["article"].str.contains(syn), "num_vocab"] += 1
        matches = small_df["article"]\
                        .str.contains(syn,
                           flags=re.IGNORECASE,
                           regex = True )
        small_df.loc[matches, "num_vocab"] += 1

        # Calculate how many times a synonym appears in an article.
        vocab_appearances = vocab_appearances.append(
                                pd.DataFrame({"syn" : syn,
                                            "matches" : matches.sum()},
                                            index = [0])
                                            
                                )

    vocab_appearances.sort_values("matches", ascending = False)
    
    merged_df = vocab_appearances\
                    .merge(gre_syn, on = "syn")
    matches_df = merged_df\
                    .groupby("word", as_index= False)\
                    .agg(matches = pd.NamedAgg("matches", "sum"))\
                    .sort_values("matches", ascending = False)
    matches_df.query("matches > 0")
    merged_df.query("word =='vacuous'")