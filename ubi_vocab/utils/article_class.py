from pydantic import BaseModel, validator
from typing import List, Optional

class ReplacedContext(BaseModel):
    vocab : Optional[str]
    syn: Optional[str]
    orig_text: Optional[List[str]]
    mod_text: Optional[List[str]]
    
    @validator('orig_text', "mod_text")
    def list_to_str(cls, v):
        if isinstance(v, list):
            v = ". ".join(v)
        return v

class Article(BaseModel):
    original: str 
    modified: str
    context : List[ReplacedContext]

