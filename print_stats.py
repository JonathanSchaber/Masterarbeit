# coding: utf-8
import matplotlib.pyplot as plt


def make_pies(labels, values, font_size, title):
    fig, ax = plt.subplots()
    a, b, c, = ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    [ _.set_fontsize(font_size) for _ in b ]
    plt.title(title, bbox={'facecolor':'0.8', 'pad':5})
    plt.show()
    
