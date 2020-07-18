

#import tensorflow as tf
import nltk

from liir.dame.core.representation.Sentence import Sentence
from liir.dame.core.representation.Text import Text
from liir.dame.core.representation.Word import Word
from liir.dame.core.representation.Predicate import Predicate
#from liir.dame.srl.DSRL import DSRL

import parzu_class as parzu


def create_ParZu_parser():
    """Create an ParZu-parser object
    Args:
        None
    Returns:
        ParZu-parser object
    """
    options = parzu.process_arguments()
    ParZu = parzu.Parser(options)
    return ParZu


def parse_sentence(parser, text):
    """parse sentence and return list of tuples with token and POS-tags
    Args:
        param1: parser Object
        param2: sentence to parse
    Returns:
        list of tuples of strings
    """
    tagged_sent = parser.tag([text])[0]
    splitted_sents = tagged_sent.split("\n")
    tagged_tuple_list = [(token.split("\t")[0], token.split("\t")[1]) for token in splitted_sents]
    return tagged_tuple_list


def create_dsrl_repr(sentences):
    """Wraps the POS-labelled tokens into a DSRL class
    Args:
        param1: list of lists of 2-tuples of strings [ (<token>, <POS>), ... ]
    Returns:
        DSRL class Text()
    """
    dsrl_text = Text()

    for sentence in sentences:
        dsrl_sentence = Sentence([Word(tuple[0]) if tuple[1] != "VVFIN" else Predicate(Word(tuple[0])) for tuple in sentence])
        dsrl_text.append(dsrl_sentence)
    
    for sentence in dsrl_text:
        for predicate in sentence.get_predicates():
            predicate.arguments = ["**UNK**" for word in sentence] 

    return dsrl_text


def main(text):
    ParZu_parser = create_ParZu_parser()
    parsed_text = parse_sentence(ParZu_parser, text)
    dsrl_text = create_dsrl_repr([parsed_text])
    import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    main("Gestern sah ich einen Vogel über die Dächer fliegen.")
    
