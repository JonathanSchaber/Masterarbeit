#import tensorflow as tf

from liir.dame.core.representation.Sentence import Sentence
from liir.dame.core.representation.Text import Text
from liir.dame.core.representation.Word import Word
from liir.dame.core.representation.Predicate import Predicate
from liir.dame.srl.DSRL import DSRL

import parzu_class as parzu

# Was ist mit VVIMP VAFIN (ev. checks einbauen?)
verb_fin_POS = ["VVFIN", "VVIMP"]

verb_inf_POS = ["VVINF", "VAINF", "VMINF", "VVPP", "VAPP", "VMPP", "VVIZU"]

verb_aux_POS = ["VAFIN", "VMFIN", "VAIMP", "VMIMP"]

subclause_marker = ["subjc", "objc", "rel"]


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


def check_if_end_note(verb, sentence):
    """return True if verb is end note, else False.
    Args:
        param1: list of str
        param2: list of lists of strs
    Returns:
        Boolean
    """
    FLAG = True
    for token in sentence:
        if token[3] == "V" and token[6] == verb[0] and not token[7] in subclause_marker:
            FLAG = False
            break
    return FLAG

def parse_text(parser, text):
    """parse sentence and return list of tuples with token and POS-tags
    Args:
        param1: parser Object
        param2: sentence to parse
    Returns:
        list of lists of tuples of strings
    """
    tagged_tuple_list = []

    sents = [sentence.rstrip().split("\n") for sentence in parser.main(text)]
    full_tagged_text = []
    for sentence in sents:
        full_tagged_text.append([token.split("\t") for token in sentence])

    for sentence in full_tagged_text:
        srl_sentence = []
        for token in sentence:
            if token[3] != "V":
                srl_sentence.append((token[1], "NOT_PRED"))
            elif token[4] in verb_fin_POS:
                srl_sentence.append((token[1], "PRED"))
            elif token[4] in verb_aux_POS:
                srl_sentence.append((token[1], "PRED")) if check_if_end_note(token, sentence) else srl_sentence.append((token[1], "NOT_PRED"))
            elif token[4] in verb_inf_POS:
                srl_sentence.append((token[1], "PRED")) if check_if_end_note(token, sentence) else srl_sentence.append((token[1], "NOT_PRED"))
            else:
                srl_sentence.append((token[1], "NOT_PRED"))
        tagged_tuple_list.append(srl_sentence)

    return tagged_tuple_list

#    sentences = nltk.sent_tokenize(text, language="german")
#    for sentence in sentences:
#        sentence = " ".join(nltk.word_tokenize(sentence, language="german"))
#        tagged_sent = parser.tag([sentence])[0]
#        splitted_sents = tagged_sent.split("\n")
#        tagged_tuple_list.append([(token.split("\t")[0], token.split("\t")[1]) for token in splitted_sents])
#    return tagged_tuple_list


def create_dsrl_repr(sentences):
    """Wraps the POS-labelled tokens into a DSRL class
    Args:
        param1: list of lists of 2-tuples of strings [ (<token>, <POS>), ... ]
    Returns:
        DSRL class Text()
    """
    dsrl_text = Text()

    for sentence in sentences:
        dsrl_sentence = Sentence([Word(tuple[0]) if tuple[1] != "PRED" else Predicate(Word(tuple[0])) for tuple in sentence])
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
    try:
        sem_roles = dsrl.predict(dsrl_obj)
    except IndexError:
        for i in range(len(dsrl_obj)):
            srl_list.append([["O"]*len(dsrl_obj[i])])
        return srl_list

    for sent in sem_roles:
        sent_list = []
        if sent.get_predicates() == []:
            sent_list.append(["O"]*len(sent))
        else:
            for predicate in sent.get_predicates():
                sent_list.append(predicate.arguments) 
        srl_list.append(sent_list)
    return srl_list


def pretty_print(dsrl, parser, text):
    """prints pretty predicted semantic roles for a given sentence.
    Args:
        param1: list of lists of lists of strings
        param2: list of lists of tuples of strings
    Returns:
        None
    """
    srl_list = predict_semRoles(dsrl, process_text(parser, text))
    parsed_text = parse_text(parser, text)
    pretty_print_list = []
    for i, sentence in enumerate(parsed_text):
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
    pretty_print(dsrl, ParZu_parser, text)
    #import pdb; pdb.set_trace()
    

if __name__ == "__main__":
    main("Einer von Tanaghrissons Männern erzählte Contrecoeur, dass Jumonville durch britisches Musketenfeuer getötet worden war.")
    
