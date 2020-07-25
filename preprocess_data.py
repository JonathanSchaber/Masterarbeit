from SemRoleLabeler import *

def preprocess_MLQA_v1(json, dsrl, parser):
    for i in range(len(json["data"])):
        for j in range(len(json["data"][i]["paragraphs"])):
            srl_context = []
            for sent in  nltk.sent_tokenize(json["data"][i]["paragraphs"][j]["context"], language="german"):
                print("=====================================\n")
                print(sent)
                srl_sent = predict_semRoles(dsrl, process_text(parser, sent))
                print(srl_sent)
                print("=====================================\n\n\n")
                if len(srl_sent) > 1:
                    raise Exception
                srl_context.append(srl_sent[0][0])
            json["data"][i]["paragraphs"][j]["srl_context"] = srl_context
