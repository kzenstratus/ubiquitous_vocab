# Load in a transformer
# And use it to compare vectorized tokens.

# Code will be similar to UKPLab implementation of huggingface.
# https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/models/Transformer.py

# Replacement can be treated also as a masking task?
# Perhaps we can perform masking task?

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional, Union, Tuple

from logger_utils import make_logger

# For fine tuning
# "https://github.com/huggingface/transformers/blob/master/examples/language-modeling/run_mlm.py"


class ST:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = "bert-base-nli-mean-tokens"
        self.model_name: str = model_name
        self._get_embedder(model_name=model_name)

    def _get_embedder(self, model_name):
        log = make_logger(__name__)
        if model_name is None:
            model_name = self.model_name

        log.info(f"Reading in pretrained model {model_name}")
        self.embedder = SentenceTransformer(model_name)
        return self.embedder

    def encode(self, text=List[str], show_progress=True, num_workers=8, save_file=None):
        log = make_logger(__name__)
        # Read in presaved embedding
        if save_file is not None and os.path.exists(save_file):
            log.info(f"Reading in pre-saved file {save_file}")
            return np.load(save_file)

        # Encoding
        log.info(
            f"Starting to embed {len(text)} text with {self.model_name}, workers : {num_workers}"
        )
        corpus_embeddings = self.embedder.encode(
            text, show_progress_bar=show_progress, num_workers=num_workers
        )

        if save_file is not None:
            np.save(save_file, np.array(corpus_embeddings))

        return corpus_embeddings


def calc_cos_sim(vec_a, vec_b):
    # norm is the size of each vector, ie. [1,0] has size 1, and [1,1] has sqrt(2) norm.
    # Angle between two vectors can be represented as a dot product.
    # Cosine similarity isn't the degree between the two vector, rather it's between [-1,1]
    # where 90 deg -> 0, 180 -> -1 similarity.
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


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
