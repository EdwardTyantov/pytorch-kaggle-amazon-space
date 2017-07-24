#-*- coding: utf8 -*-


def posmin(seq, key=lambda x: x):
    return min(enumerate(seq), key=lambda k: key(k[1]))[0]


def posmax(seq, key=lambda x: x):
    return max(enumerate(seq), key=lambda k: key(k[1]))[0]
