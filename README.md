# ubiquitous_vocab
GTech Ed Tech Project 

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/kzenstratus/ubiquitous_vocab/main?filepath=ubi_vocab%2Futils)

# Summary:
Building long term vocabulary is difficult. This project replaces words found in every day news articles with words that you are trying to learn. At the moment, this project is mainly focused around the backend tech that will support accurate vocabulary replacements.


## Research has shown that a very effective way of learning vocabulary for long term retention:
* Frequent exposure
* Over long spaced out long periods of time
* Through context (reading).

However this environment doesn't occur naturally.

## Solution
The ultimate outcome of this overall project will be a tool, such as a web extension, which will modify the digital text that someone is reading, to include vocabulary words that they are attempting to study. 

I plan to focus most of my efforts in developing the underlying back end technology, rather than the UI (front end components, chrome extension, server hosting work, etc). 

I will work on the technology that will take a large input text (news article length), some vocabulary words, definitions, and examples, and insert those words into the original input text without losing the meaning of the sentence. 


# Example:
* Click on the above binder badge for an interactive notebook.

# Current Technology:
* spacy for POS matching.
* wordnet for word sense disambiguation and pos matching.
* LESK for word sense disambiguation.
* BERT for word sense disambiguation.


# Data:
* News article data - https://components.one/datasets/all-the-news-2-news-articles-dataset/

* GRE data from - https://github.com/ParasAvkirkar/MagooshHelper

# For Developers:

Create and install dependencies into a conda environment
```
$ make setup
```

