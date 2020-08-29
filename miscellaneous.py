# coding: utf-8
import matplotlib.pyplot as plt


def make_pies(labels, values, colors, font_size=10, title=None):
    fig, ax = plt.subplots()
    a, b, _, = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    [ _.set_fontsize(font_size) for _ in b ]
    if title:
        plt.title(title, bbox={'facecolor':'0.8', 'pad':5})
    plt.show()
    
def make_pies(labels, values, colors, font_size=10, title=None):
      fig, ax = plt.subplots()
      a, b, _, = ax.pie(values, colors=colors, pctdistance=0.6, autopct='%1.1f%%', startangle=90)
      [ _.set_fontsize(font_size) for _ in b ]
      ax.legend(a, labels, loc="center right", bbox_to_anchor=(1.7, 0.5), fontsize="x-large")
      if title:
          plt.title(title, bbox={'facecolor':'0.9', 'pad':5})
      plt.show()

# def make_nested_pies(labels_out, colors_out, values_out, labels_in, colors_in, values_in):
#     fig, ax = plt.subplots()
#     ax.axis('equal')
#     width = 0.3
#     pie, _, _ = ax.pie(values_out, radius=1, labels=labels_out, colors=colors_out, autopct='%1.1f%%', startangle=90)
#     plt.setp( pie, width=width, edgecolor='white')
#     pie2, _, _ = ax.pie(values_in, radius=1-width, labels=labels_in, colors=colors_in, autopct='%1.1f%%', startangle=90)
#     plt.setp( pie2, width=width, edgecolor='white')
#     plt.show()
# 


def vals(data):
    """function to compute length of strings for QA data sets
    Args:
        param1: list[list]
    """
     answer_normal = 0
     question_normal = 0
     context_normal = 0
     answer_token = 0
     question_token = 0
     context_token = 0
     for x in data:
         answer_token += len(tokenizer.tokenize(x[1]))
         question_token += len(tokenizer.tokenize(x[3]))
         context_token += len(tokenizer.tokenize(x[2]))
         answer_normal += len([y for y in tokenizer.tokenize(x[1]) if not y.startswith("##")])
         question_normal += len([y for y in tokenizer.tokenize(x[3]) if not y.startswith("##")])
         context_normal += len([y for y in tokenizer.tokenize(x[2]) if not y.startswith("##")])
     print("av. length answer: {:.1f} ({:.1f})".format(answer_normal/len(data), answer_token/len(data)))
     print("av. length question: {:.1f} ({:.1f})".format(question_normal/len(data), question_token/len(data)))

     print("av. length context: {:.1f} ({:.1f})".format(context_normal/len(data), context_token/len(data)))
