import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag

lemmatizer = nltk.stem.WordNetLemmatizer()

def filter_user(tweets):
    """Removes <user> tags from tweets by replacing them with empty strings
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with <user> tags removed
    """
    
    return tweets.str.replace('<user>', '', case=False)


def filter_url(tweets):
    """Removes <url> tags from tweets by replacing them with empty strings
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with <url> tags removed
    """
    
    return tweets.str.replace('<url>', '', case=False)


def filter_rt(tweets):
    """Removes 'rt' words from tweets by replacing them with empty strings
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with 'rt' words removed
    """
    
    return tweets.str.replace('^rt ', ' ', case=False)


def filter_comma(tweets):
    """Removes commas from tweets by replacing them with empty strings
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with commas removed
    """
    
    return tweets.str.replace(' , ', ' ', case=False)


def filter_multi_whitespace(tweets):
    """Removes multiple whitespace occurrences from tweets by replacing them with a single whitespace
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with multiple whitespaces removed
    """
    
    # Remove whitespaces at the beginning of the tweet
    tweets = tweets.str.replace(r'^[ ]+', '', case=False)
    
    # Replace multiple whitespaces elsewhere with a single whitespace
    tweets = tweets.str.replace(r'[ ]{2,}', ' ', case=False)
    
    return tweets

def filter_emojis(tweets):
    """Removes common emojis from tweets by replacing them with tags
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with emojis replaced
    """
    # Replace emojis with tags
    tweets = tweets.str.replace(r'<3', ' <love> ', case=False)
    tweets = tweets.str.replace(r' :\* ', ' <kiss> ', case=False)
    tweets = tweets.str.replace(r' :d ', ' <laughface> ', case=False)
    tweets = tweets.str.replace(r' ;d ', ' <laughface> ', case=False)
    tweets = tweets.str.replace(r' :p ', ' <laughface> ', case=False)
    tweets = tweets.str.replace(r' ;p ', ' <laughface> ', case=False)
    tweets = tweets.str.replace(r'[:=][\']?[\-]?[\)\]]', ' <smile> ', case=False)
    tweets = tweets.str.replace(r'\b[l]+[m]+?[a]+?[o]+\b', '<laugh>', case=False)
    tweets = tweets.str.replace(r' : > ', ' <smile> ', case=False)
    tweets = tweets.str.replace(r'\b[l]+[o]+?[l]+?[s]*\b', ' <lolexpr> ', case=False)
    tweets = tweets.str.replace(r'[:=]/', ' <sadface> ', case=False)
    tweets = tweets.str.replace(r'[:=][\']?[\-]?[\)\[]', ' <sadface> ', case=False)
    tweets = tweets.str.replace(r'\b > . <', '<sadface>', case=False)
    tweets = tweets.str.replace(r'\b[o]+[m]+?[g]+?\b', ' <omg> ', case=False)
    
    return tweets


def filter_numbers(tweet):
    """Filters numbers from a tweet and replaces them with <number>
    
    Args: 
        tweet (string)
    Returns: 
        tweet (string) : tweet with numbers removed
    """
    out = []
    for token in tweet.split():
        stripped = re.sub('[^a-zA-Z0-9 \n]', '', token)
        try:
            int(stripped)
            out.append('<number>')
        except:
            out.append(token)
    return ' '.join(out).strip()

def filter_words_with_numbers(tweets):
    """Removes words with numbers from tweets (e.g. 1050vp4, 75xt036aa)
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets without words with numbers
    """
        
    return tweets.str.replace(r'((\w+)?\d+(\w+)?)','', case=False)

def filter_repeated_puctuation(tweets):
    """Filters numbers from a tweet and replaces them with <number>
    
    Args: 
        tweet (string)
    Returns: 
        tweet (string) : tweet with numbers removed
    """
    return tweets.str.replace(r'([?!]\s?)\1+', r'\1' + ' <emphasize> ', case=False)

def filter_kisses(tweets):
    """Replace variations of xx with <kisses>
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with multiple whitespaces removed
    """
    tweets = tweets.str.replace(r'\b([x])\1+\b', ' <kisses> ', case=False)
    tweets = tweets.str.replace(r'[o]?([x][o])\1+', ' <kisses> ', case=False)
    
    return tweets


def filter_haha(tweets):
    """Merges different variations of laughs (e.g. haha, ahah, hahaha, ahahahah, etc.)
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with laughs replaced with <lolexpr>
    """

    return tweets.str.replace('haha[ha]*|ahah[ah]*', ' <lolexpr> ', case=False)   

def filter_single_characters(tweets):
    """Filters single characters and (possibly repeated) full stops from a tweet.
    
    Args: 
        tweet (string)
    Returns: 
        tweet (string) : tweet with numbers removed
    """
    tweets = tweets.str.replace(r'(\s[^?!](\s[^?!])*\s)', ' ', case=False)
    tweets = tweets.str.replace(r'(\s[^?!]$)', '', case=False) # end
    tweets = tweets.str.replace(r'(^[^?!]\s)', '', case=False) # start
    tweets = tweets.str.replace(r'(\.\.$)', '', case=False) # dots at the and
    
    return tweets

def filter_extra_letters(tweets): 
    """Function to replace multiple [3+ times] occurrences   
    of a character by a single character
    """ 
    
    return tweets.str.replace(r'(.)\1+', r'\1\1', case=False)


def expand_contractions(tweets):
    """Expands contractions that exist in informal writing (e.g. isn't --> is not) in order to recover their sentiment
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with contractions expanded 
    """
    
    contractions = {
        
        # Handle cases with apostrophe:
        
        'won\'t': 'will not',
        'can\'t': 'cannot',
        'ain\'t': 'is not',
        'n\'t': ' not',
        'i\'m': 'i am',
        'he\'s': 'he is',
        'it\'s': 'it is',
        '\'re': ' are',
        'that\'s': 'that is',
        'what\'s': 'what is',
        'where\'s': 'where is',
        'who\'s': 'who is',
        'when\'s': 'when is',
        'why\'s': 'why is',
        'how\'s': 'how is',
        '\'ll': ' will',
        '\'ve': ' have',
        '\'d': ' would',
        '\'s': '',
        's\'': '',
        
        # Handle cases without apostrophe:
        
        'arent': 'are not',
        'wont': 'will not',
        'wasnt': 'was not',
        'isnt': 'is not',
        'werent': 'were not',
        'cant': 'cannot',
        'aint': 'is not',
        'dont': 'do not',
        'didnt': 'did not',
        'couldnt': 'could not',
        'shouldnt': 'should not',
        'wouldnt': 'would not',
        'havent': 'have not',
        'im': 'i am',
        '\b hes \b': 'he is',
        '\b ive \b': 'i have',
        'youre': 'you are',
        'youve': 'you have',
        'dont': 'do not',
        '\b weve \b': 'we have',
        'theyre': 'they are',
        'theyve': 'they have',
        'dunno': 'do not know',
        ' ill ': ' i will ',
        'youll': 'you will',
        'theyll': 'they will',
    }
    
    # For all tweets, match the pattern key in contractions dictionary and replace with its value
    pattern_obj = re.compile('|'.join(contractions.keys()))
    tweets  = tweets.apply(lambda tweet: pattern_obj.sub(lambda contr: contractions[contr.group()], tweet))
    
    return tweets

def filter_hashtags(tweets):
    """Removes hashtag marks from tweets by replacing them with tags
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets with hashtags replaced
    """

    return tweets.str.replace('#', ' <hashtag> ', case=False)

def lemmatize(tweet):
    """Lemmatizes a tweet (e.g. tried -> try)
    Args: 
        tweet (string)
    Returns: 
        tweet (string) : tweet where the words have been lemmatized
    """
    out = []
    for word, tag in pos_tag(tweet.split()):
        t = tag[0].lower() if tag[0] in ['A', 'R', 'N', 'V'] else None
        if t == None:
            lemma = word
        else:
            lemma = lemmatizer.lemmatize(word, t)
        out.append(lemma)
        
    return ' '.join(out)  

def drop_stopwords(tweets):
    """Removes stopwords from tweets except where they might indicate sentiment.
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : strings of tweets without stopwords
    """
    
    stops = stopwords.words("english")
    stops = [word for word in stops if not bool(re.match(r'(\w+)n\'t|(\w+)[dts]n', word))]
    non_stop = ['against', 'up', 'down', 'no', 'nor', 'very', 'only', 'not', 'ain', 'aren', 
                'haven', 'isn', 'don', 'can', 'weren', 'shan', 'won', 'should\'ve']
    stops = list(set(stops) - set(non_stop))
    
    return tweets.apply(lambda tweet: ' '.join([w for w in tweet.split() if not w.lower() in stops]))

def preprocess_data(tweets):
    """Calls appropriate functions to preprocess tweets.
    
    Args:
        tweets (pandas series) : strings of tweets
    Returns: 
        tweets (pandas series) : preprocessed strings of tweets 
    """
        
    tweets = filter_user(tweets)
    tweets = filter_url(tweets)
    tweets = filter_rt(tweets)
    tweets = filter_hashtags(tweets)
    tweets = filter_emojis(tweets)
    tweets = filter_comma(tweets)
    tweets = filter_extra_letters(tweets)
    tweets = filter_haha(tweets)
    tweets = expand_contractions(tweets)
    tweets = filter_kisses(tweets)
    tweets = tweets.apply(lambda tweet: lemmatize(tweet))
    tweets = drop_stopwords(tweets)
    tweets = filter_repeated_puctuation(tweets)
    tweets = tweets.apply(lambda tweet: filter_numbers(tweet)) # run after emoji filtering
    tweets = filter_words_with_numbers(tweets) # needs to run after number filtering
    tweets = filter_single_characters(tweets) # run last
    tweets = filter_multi_whitespace(tweets)
    
    return tweets
