from nrclex import NRCLex
import numpy as np
import pandas as pd

def multiCategorySentiment(text):
    text_object = NRCLex(text)
    # countDict = dict({
    #     'fear': 0, 
    #     'sadness': 0, 
    #     'negative': 0, 
    #     'disgust': 0, 
    #     'anticip':0, 
    #     'joy': 0,
    #     'trust': 0,
    #     'positive': 0,
    #     'surprise': 0,
    #     'anger': 0
    #     })
    # countDict.update(text_object.raw_emotion_scores)
    # total = sum(countDict.values())
    # if total == 0:
    #     return countDict
    # norm = {k: v/total for k, v in countDict.items()}
    return text_object.affect_frequencies #norm

def pandasMultiCategorySentiment(text):
    norm = multiCategorySentiment(text)
    return pd.Series([norm['fear'], norm['sadness'], norm['negative'], norm['disgust'], norm['anticip'], norm['joy'], norm['trust'], norm['positive'], norm['surprise'], norm['anger']])