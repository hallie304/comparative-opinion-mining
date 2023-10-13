import numpy as np
import copy
import torch

# clear the string of special symbol.
def clear_string(line, strip_symbol=None, replace_symbol=None):
    """
    :param line: a string
    :param strip_symbol:
    :param replace_symbol: a list of special symbol, need replace.
    :return:
    """
    if strip_symbol is not None:
        for sym in strip_symbol:
            line = line.strip(sym)

    if replace_symbol is not None:
        for sym in replace_symbol:
            line = line.replace(sym, "")

    return line

# using split symbol get a list of string.
def split_string(line, split_symbol):
    """
    :param line: a string need be split
    :param split_symbol: a string: split symbol
    :return:
    """
    return list(filter(None, line.split(split_symbol)))

# read file to get sentence and label
def read_standard_file(path):
    """
    :param path:
    :return: sent_col, sent_label_col and label_col
    """
    sent_col, sent_label_col, final_label_col = [], [], []
    last_sentence = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip('\n')

            # "[[" denote the begin of sequence label.
            if line[:2] == "[[":
                label_col.append(line)

            else:
                if last_sentence != "":
                    cur_sent, cur_sent_label = split_string(last_sentence, "\t")
                    sent_col.append(cur_sent)
                    sent_label_col.append(int(cur_sent_label))
                    final_label_col.append(label_col)

                last_sentence = clear_string(line, replace_symbol={u'\u3000': u""})
                label_col = []

        cur_sent, cur_sent_label = split_string(last_sentence, "\t")
        sent_col.append(cur_sent)
        sent_label_col.append(int(cur_sent_label))
        final_label_col.append(label_col)

        return sent_col, sent_label_col, final_label_col