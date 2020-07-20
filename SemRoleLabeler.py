#import tensorflow as tf
import nltk

from liir.dame.core.representation.Sentence import Sentence
from liir.dame.core.representation.Text import Text
from liir.dame.core.representation.Word import Word
from liir.dame.core.representation.Predicate import Predicate
from liir.dame.srl.DSRL import DSRL

import parzu_class as parzu

# Was ist mit VVIMP VAFIN (ev. checks einbauen?)
verb_fin_POS = ["VVFIN", "VVIMP"]

verb_inf_POS = ["VVINF", "VVPP"]

aux_verb_POS = ["VAFIN", "VMFIN"]


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


def parse_text(parser, text):
    """parse sentence and return list of tuples with token and POS-tags
    Args:
        param1: parser Object
        param2: sentence to parse
    Returns:
        list of lists of tuples of strings
    """
    tagged_tuple_list = []
    full_tagged_list = []

    sentences = nltk.sent_tokenize(text, language="german")
    for sentence in sentences:
        sentence = " ".join(nltk.word_tokenize(sentence, language="german"))
        tagged_sent = parser.tag([sentence])[0]
        splitted_sents = tagged_sent.split("\n")
        tagged_tuple_list.append([(token.split("\t")[0], token.split("\t")[1]) for token in splitted_sents])
        # full_tagged_list.append([token.split("\] 
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
        AUX_FLAG = True if aux_verb_POS in [tuple[1] for tuple in sentence]

        dsrl_sentence = Sentence([Word(tuple[0]) if tuple[1] not in verb_POS else Predicate(Word(tuple[0])) for tuple in sentence])
        dsrl_text.append(dsrl_sentence)
    
    for sentence in dsrl_text:
        for predicate in sentence.get_predicates():
            predicate.arguments = ["**UNK**" for word in sentence] 

    return dsrl_text


def process_text(parser, text):
    """take raw sentence and retunrn DSRL-processable object
    Args:
        param1: ParZu parser Object
        param2: str
    Returns:
        DSRL class Text()
    """
    parsed_text = parse_text(parser, text)
    dsrl_text_object = create_dsrl_repr(parsed_text)
    return dsrl_text_object


def predict_semRoles(dsrl, dsrl_obj):
    """returns list of lists of SRL-tag per token
    Args:
        param1: DAMESRL DSRL() object
        param2: DAMESL Text() object
    Returns:
        list of lists of str
    """
    srl_list = []
    sem_roles = dsrl.predict(dsrl_obj)
    for sent in sem_roles:
        sent_list = []
        for predicate in sent.get_predicates():
           sent_list.append(predicate.arguments) 
           if not predicate:
               sent_list.append([])
        srl_list.append(sent_list)
    return srl_list


def pretty_print(srl_list, parse_text):
    """prints pretty predicted semantic roles for a given sentence.
    Args:
        param1: list of lists of lists of strings
        param2: list of lists of tuples of strings
    Returns:
        None
    """
    pretty_print_list = []
    for i, sentence in enumerate(parse_text):
        for j, token in enumerate(sentence):
            pretty_print_list.append("\t".join(token) + "\t" + "\t".join(semrole_item[j] for semrole_item in srl_list[i]))
        pretty_print_list.append("\n")
    print("\n".join(pretty_print_list))


def main(text):
    argument_model_config = "../SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini"
    dsrl = DSRL(argument_model_config)
    ParZu_parser = create_ParZu_parser()
    dsrl_obj = process_text(ParZu_parser, text)
    sem_roles = predict_semRoles(dsrl, dsrl_obj)
    pretty_print(sem_roles, parse_text(ParZu_parser, text))
    import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    main("Wer hat die Gans gestohlen? Gib sie wieder her! Ich will das nicht sehen müssen.")
    
