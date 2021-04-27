from typing import List, Optional, Any

from transformer import ST, calc_cos_sim


class ReplacedContext(BaseModel):
    """Info around a replaced vocabulary.

    Examples:
        >>> rc = ReplacedContext(vocab = "detrimental",
                                     syn = "damaging",
                                     orig_text = ["damaging to health"],
                                     mod_text = ["detrimental to health"])
    """

    vocab: Optional[str]  # target vocabulary word
    syn: Optional[str]  # synonym of target vocabulary word.
    orig_text: Optional[List[str]]  # original text with syn.
    mod_text: Optional[List[str]]  # text modified with syn to be replaced by vocab.
    similarity_score: Optional[List[float]]

    # @validator('orig_text', "mod_text")
    # def list_to_str(cls, v):
    #     if isinstance(v, list):
    #         v = ". ".join(v)
    #     return v

class Article(BaseModel):
    original: str
    modified: str
    context: List[ReplacedContext]

