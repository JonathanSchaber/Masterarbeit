#import tensorflow as tf

from liir.dame.core.representation.Predicate import Predicate
from liir.dame.core.representation.Sentence import Sentence
from liir.dame.core.representation.Text import Text
from liir.dame.core.representation.Word import Word
from liir.dame.srl.DSRL import DSRL

import parzu_class as parzu


class SRL_predictor:
    def __init__(self, argument_model_config):
        self.dsrl = DSRL(argument_model_config)
        self.parser = None
        # Was ist mit VVIMP VAFIN (ev. checks einbauen?)
        self.verb_fin_POS = ["VVFIN", "VVIMP"]
        self.verb_inf_POS = ["VVINF", "VAINF", "VMINF", "VVPP", "VAPP", "VMPP", "VVIZU"]
        self.verb_aux_POS = ["VAFIN", "VMFIN", "VAIMP", "VMIMP"]
        self.subclause_marker = ["subjc", "objc", "rel"]
        self.create_ParZu_parser()

    @staticmethod
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

    def create_ParZu_parser(self):
        """Create an ParZu-parser object
        Args:
            None
        Returns:
            ParZu-parser object
        """
        options = parzu.process_arguments(commandline=False)
        ParZu = parzu.Parser(options)
        self.parser = ParZu

    def check_if_end_note(self, verb, sentence):
        """return True if verb is end note, else False.
        Args:
            param1: list of str
            param2: list of lists of strs
        Returns:
            Boolean
        """
        FLAG = True
        for token in sentence:
            if token[3] == "V" and token[6] == verb[0] and not token[7] in self.subclause_marker:
                FLAG = False
                break
        return FLAG

    def parse_text(self, text):
        """parse sentence and return list of tuples with token and POS-tags
        Args:
            param1: parser Object
            param2: sentence to parse
        Returns:
            list of lists of tuples of strings
        """
        tagged_tuple_list = []

        sents = [sentence.rstrip().split("\n") for sentence in self.parser.main(text)]
        full_tagged_text = []
        for sentence in sents:
            full_tagged_text.append([token.split("\t") for token in sentence])

        for sentence in full_tagged_text:
            srl_sentence = []
            for token in sentence:
                if token[3] != "V":
                    srl_sentence.append((token[1], "NOT_PRED"))
                elif token[4] in self.verb_fin_POS:
                    srl_sentence.append((token[1], "PRED"))
                elif token[4] in self.verb_aux_POS:
                    if self.check_if_end_note(token, sentence):
                        srl_sentence.append((token[1], "PRED"))
                    else:
                        srl_sentence.append((token[1], "NOT_PRED"))
                elif token[4] in self.verb_inf_POS:
                    if self.check_if_end_note(token, sentence):
                        srl_sentence.append((token[1], "PRED"))
                    else:
                        srl_sentence.append((token[1], "NOT_PRED"))
                else:
                    srl_sentence.append((token[1], "NOT_PRED"))
            tagged_tuple_list.append(srl_sentence)

        return tagged_tuple_list

    def predict_semRoles(self, text):
        """returns list of lists of SRL-tag per token
        Args:
            param1: DAMESRL DSRL() object
            param2: DAMESL Text() object
        Returns:
            list of lists of str
        """
        srl_list = []

        parsed_text = self.parse_text(text)
        dsrl_obj = self.create_dsrl_repr(parsed_text)

        try:
            sem_roles = self.dsrl.predict(dsrl_obj)
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


    def pretty_print(self, text):
        """prints pretty predicted semantic roles for a given sentence.
        Args:
            param1: list of lists of lists of strings
            param2: list of lists of tuples of strings
        Returns:
            None
        """
        srl_list = predict_semRoles(self.dsrl, process_text(self.parser, text))
        parsed_text = parse_text(self.parser, text)
        pretty_print_list = []
        for i, sentence in enumerate(parsed_text):
            for j, token in enumerate(sentence):
                pretty_print_list.append("\t".join(token) + "\t" + "\t".join(semrole_item[j] for semrole_item in srl_list[i]))
            pretty_print_list.append("\n")
        print("\n".join(pretty_print_list))
