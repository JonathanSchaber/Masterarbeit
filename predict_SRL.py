#import tensorflow as tf
import nltk.data

from liir.dame.core.representation.Predicate import Predicate
from liir.dame.core.representation.Sentence import Sentence
from liir.dame.core.representation.Text import Text
from liir.dame.core.representation.Word import Word
from liir.dame.srl.DSRL import DSRL

from transformers import BertTokenizer

import parzu_class as parzu


# /home/joni/Documents/Uni/Master/Computerlinguistik/20HS_Masterarbeit/SemRolLab/DAMESRL/server_configs/srl_char_att_ger_infer.ini


class SRL_predictor:
    def __init__(self, argument_model_config, bert_path):
        self.dsrl = DSRL(argument_model_config)
        self.tokenizer = BertTokenizer.from_pretrained(bert_path)
        self.parser = None
        # Was ist mit VVIMP VAFIN (ev. checks einbauen?)
        self.verb_fin_POS = ["VVFIN", "VVIMP"]
        self.verb_inf_POS = ["VVINF", "VAINF", "VMINF", "VVPP", "VAPP", "VMPP", "VVIZU"]
        self.verb_aux_POS = ["VAFIN", "VMFIN", "VAIMP", "VMIMP"]
        self.subclause_marker = ["subjc", "objc", "rel"]
        self.sent_detector = nltk.data.load('tokenizers/punkt/german.pickle')
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
            dsrl_sentence = Sentence(
                                [Word(tuple[0])
                                    if tuple[1] != "PRED"
                                    else Predicate(Word(tuple[0])) 
                                    for tuple in sentence]
                                )
            dsrl_text.append(dsrl_sentence)
        
        for sentence in dsrl_text:
            for predicate in sentence.get_predicates():
                predicate.arguments = ["**UNK**" for word in sentence] 

        return dsrl_text

    @staticmethod
    def merge_subtokens(sentence):
        merged_sent = []
        for i, token in enumerate(sentence):
            if token.startswith("##"):
                continue
            elif i+1 == len(sentence):
                merged_sent.append(token)
            elif not sentence[i+1].startswith("##"):
                merged_sent.append(token)
            else:
                current_word = [token]
                for subtoken in sentence[i+1:]:
                    if subtoken.startswith("##"):
                        current_word.append(subtoken.lstrip("##"))
                    else:
                        break
                merged_sent.append("".join(current_word))
        return merged_sent

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
            param1: text to parse
        Returns:
            list of lists of tuples of strings
        """
        tagged_tuple_list = []

        bert_full_text = self.merge_subtokens(self.tokenizer.tokenize(text))

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
            #tagged_tuple_list.append(srl_sentence)

            bert_srl_sentence = []

            for token_label in srl_sentence:
                for subtoken in self.merge_subtokens(self.tokenizer.tokenize(token_label[0])):
                    bert_srl_sentence.append((subtoken, token_label[1]))

            tagged_tuple_list.append(bert_srl_sentence)

        offset = 0
        if not  sum([len(sent) for sent in tagged_tuple_list]) == len(bert_full_text):
            bert_sents = self.sent_detector.tokenize(text)
            assert len(tagged_tuple_list) == len(bert_sents)
            for i, sent in enumerate(tagged_tuple_list):
                bert_sent = self.merge_subtokens(self.tokenizer.tokenize(bert_sents[i]))
                if [token[0] for token in sent] != bert_sent:
                    controlled_sent = []
                    q = lambda x: 2 if x < 2 else x
                    for j, token in enumerate(bert_sent):
                        FLAG = False
                        window = sent[q(j)-2:j+2]
                        for word, label in window:
                            if word == token and label == "PRED":
                                controlled_sent.append((word, label))
                                FLAG = True
                                break
                        if not FLAG:
                            controlled_sent.append((word, "NOT_PRED"))
                    if not len(controlled_sent) == len(bert_sent):
                        import ipdb; ipdb.set_trace()
                    tagged_tuple_list[i] = controlled_sent

                #    offset += len(sent)
                #elif len(sent) > len(bert_sent):
                #    while len(sent) != len(bert_sent):
                #        for i, token in enumerate(sent):
                #            r = lambda x: 1 if x < 1 else x
                #            q = lambda x: 2 if x < 2 else x
                #            tuple_pattern = [token[0] for token in sent[r(i)-1:i+2]]
                #            bert_window = bert_sent[q(i)-2:i+3]
                #            if len(tuple_pattern) == 2:
                #                bert_patterns = [bert_window[k:k+2] for k in range(len(bert_window))]
                #            else:
                #                bert_patterns = [bert_window[r(k)-1:k+2] for k in range(len(bert_window))]
                #            if tuple_pattern not in bert_patterns:
                #                print([x[0] for x in sent])
                #                print("after deleting: {}".format(sent[i]))
                #                del(sent[i])
                #                print([x[0] for x in sent])
                #                import ipdb; ipdb.set_trace()
                #                break
                #else:
                #    import ipdb; ipdb.set_trace()

        if not  sum([len(sent) for sent in tagged_tuple_list]) == len(bert_full_text):
            import ipdb; ipdb.set_trace()

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

        # if there are no predicates in all sentences, return empty-SRLs for
        # all sentences
        try:
            sem_roles = self.dsrl.predict(dsrl_obj)
        except IndexError:
            for i in range(len(dsrl_obj)):
                srl_list.append([["0"]*len(dsrl_obj[i])])
            return srl_list

        for sent in sem_roles:
            sent_list = []
            if sent.get_predicates() == []:
                sent_list.append(["0"]*len(sent))
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
        srl_list = self.predict_semRoles(text)
        parsed_text = self.parse_text(text)
        pretty_print_list = []

        for sent in zip(srl_list, parsed_text):
            for i, token in enumerate(sent[1]):
                print(token[0] + "\t" + "\t".join(srl[i] for srl in sent[0]))
            print("\n")
                
