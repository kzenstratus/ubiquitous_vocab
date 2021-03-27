# Use magoosh helper repo - https://github.com/ParasAvkirkar/MagooshHelper
import os, re
import pandas as pd

from logger_utils import make_logger

def parse_raw_vocab_line(s : str) -> pd.DataFrame:
    # Split out the word and pos. 
    # the definitions and examples are too noisy to cleanly parse.
    # Use NLTK for that down the line.

    name = re.sub(pattern = r"^([a-z]+) .*",
                 repl = "\\1",
                 string = s
                 ).strip()
    pos = re.sub(r"^[a-z]+ \(([a-z]+)\):.*",
                 "\\1", s).strip()

    return pd.DataFrame({'word' : name,
                        'pos' : pos},
                         index = [0])

def get_raw_vocab(out_file : str = None) -> pd.DataFrame:
    
    if out_file is not None and os.path.exists(out_file):
        print(f"Reading existing file at {out_file}")
        return pd.read_csv(out_file)
    data_dir = "/Users/kevinzen/repos/MagooshHelper"
    data_files = {"basic" : "Basic Words.csv",
                 "advanced" : "Advance Words.csv",
                 "common" : "Common Words.csv"}
    all_data = pd.DataFrame()
    # For each section read in all vocab words.
    for section, data_file in data_files.items():
        data_file = os.path.join(data_dir, data_file)
        with open(data_file, 'r') as f:
            all_vocab = pd.concat(
                            [parse_raw_vocab_line(line) for line in f])
        all_vocab["section"] = section
        all_data = all_data.append(all_vocab)
    
    if out_file is not None:
        all_data.to_csv(out_file, index = False)
    
    return all_data

def get_clean_news(news_file : str = None) -> pd.DataFrame:
    news_df = get_news(news_file = news_file)
    # Remove indexes
    cols_to_drop = [col for col in list(news_df) if bool(re.search("Unnamed", col))]
    
    news_df.drop(columns = cols_to_drop, inplace = True)
    news_df = news_df.query("article.notnull()")
    news_df.to_csv(news_file, index = False)
    return news_df
def get_news(news_file : str = None) -> pd.DataFrame:
    if news_file is None:
        news_file = "~/repos/ubiquitous_vocab/data/all-the-news-2-1.csv"
    news_df = pd.read_csv(news_file)
    
    return news_df