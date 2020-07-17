

#import tensorflow as tf
import nltk

from liir.dame.core.representation.Sentence import Sentence
from liir.dame.core.representation.Text import Text
from liir.dame.core.representation.Word import Word
from liir.dame.core.representation.Predicate import Predicate
#from liir.dame.srl.DSRL import DSRL

import parzu_class as parzu



def parse_sentence(parser: parzu_class.Parser, text: str):
    """parse sentence and return list of tuples with token and POS-tags
    Args:
        param1: parser Object
        param2: sentence to parse
    Returns:
        list of tuples of strings
    """
    


def create_dsrl_repr(text: list):
    """Wraps the POS-labelled tokens into a DSRL class
    Args:
        param1: list of 2-tuples of strings [ (<token>, <POS>), ... ]
    Returns:
        DSRL class Text()
    """
    dsrl_text = Text()

    sentence = Sentence([Word(tuple[0]) if tuple[1] != "VVFIN" else Predicate(Word(tuple[0])) for tuple in text])

    dsrl_text.append(sentence)
    
    for sentence in dsrl_text:
        for predicate in sentence.get_predicates():
            predicate.arguments = ["**UNK**" for word in sentence] 

    return dsrl_text


def main(text):
    print(create_dsrl_repr(text))

if __name__ == "__main__":
    main([("Der", "ART"), ("Hund", "NN"), ("jagt", "VVFIN"), ("die", "ART"), ("Katze", "ART"), (".", "$.")])
