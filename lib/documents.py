import spacy
from spacy.tokens import Doc
from functools import reduce
from collections import OrderedDict, deque
from typing import overload

# Constants
lemma_tags = {"NNS", "NNPS"}
punctuation = """ '".,;:!?&#$%-+/>[]{}()\\\n"""

# Some helper functions
def texthasalphanumdash(text):
    return all(c.isalnum() or c == "-" or c == "." for c in text) and any(c.isalpha() for c in text)

def filter_tokens(token):
    return not token.is_stop and texthasalphanumdash(token.text)

def singularize(token):
    return token.lemma_ if token.tag_ in lemma_tags else token.text

def singularize_chunk(chunk):
    return " ".join(
        map(singularize, filter(filter_tokens, chunk))
    ).lower().strip(punctuation)

def prepare_words(string, char):
    return string.replace(char, f" {char}")

def lowercase_first(string):
    return string[0].lower() + string[1:]

# Our custom tokenizer
class CustomTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        # Construct words from the text
        words = list(
            map(
                lowercase_first,
                reduce(
                    prepare_words,
                    [". ", ", ", "! ", "? ", ": ", "; "],
                    text
                ).split()
            )
        )

        # Create a Doc object
        return Doc(self.vocab, words=words)

class Documents:
    def __init__(self, model="en_core_web_sm"):
        self.paper_keywords = []
        self.keywords = OrderedDict()
        self.args = {"batch_size": 256, "n_process": -1, "disable": ["ner"]}

        # Load the spacy model
        self.nlp = spacy.load(model)
        self.nlp.tokenizer = CustomTokenizer(self.nlp.vocab)

    def clear(self):
        self.paper_keywords = []
        self.keywords = OrderedDict()

    @overload
    def extract_noun_chunks(self, texts: str): ...
    @overload
    def extract_noun_chunks(self, texts: list): ...

    # Override the extract_keywords method
    def extract_noun_chunks(self, texts: list | str):
        self.clear()
        if isinstance(texts, str):
            self.process_doc_noun_chunks(self.nlp(texts))
            return list(self.keywords.values())
        deque(map(self.process_doc_noun_chunks, self.nlp.pipe(texts, **self.args)))

    # Extract noun chunks
    def process_doc_noun_chunks(self, doc):
        keywords = []

        for chunk in doc.noun_chunks:
            # Add the keyword to the list
            keyword = singularize_chunk(chunk)
            if len(keyword) > 1:
                keyword_hash = doc.vocab.strings[keyword]
                keywords.append(keyword_hash)
                self.keywords[keyword_hash] = keyword
        self.paper_keywords.append(keywords)