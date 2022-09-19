import json
import string
import numpy as np
import random

np.random.seed(42)

from enum import Enum
import itertools
from random import randrange
import copy

import spacy
nlp_spacy = spacy.load("en_core_web_sm")
# merge_nps = nlp_spacy.create_pipe("merge_noun_chunks")
# nlp_spacy.add_pipe(merge_nps)
nlp_spacy.add_pipe("merge_noun_chunks")


class ShuffleType(Enum):
    UNIGRAM = 1
    BIGRAM = 2
    TRIGRAM = 3
    ONE_WORD = 4  # Randomly pick one word and insert it to another place in a sentence.


def shuffle_tokens(tokens, type=ShuffleType.UNIGRAM, strict=False):

    original_indices = np.arange(0, len(tokens))

    if type == ShuffleType.BIGRAM:
        modified_indices = [idx for idx in original_indices if idx % 2 == 0]
        shuffled_indices = np.random.permutation(modified_indices)

        # Shufle the list of tokens until there is at least a change
        while all(shuffled_indices == modified_indices) and len(shuffled_indices) > 1:
            shuffled_indices = np.random.permutation(modified_indices)

        if strict:
            while any(shuffled_indices == modified_indices) and len(shuffled_indices) > 1:
                shuffled_indices = np.random.permutation(modified_indices)

        # Reconstruct pair-wise shuffled list
        shuffled_indices = [[idx, idx+1] for idx in shuffled_indices]
        shuffled_indices = list(itertools.chain.from_iterable(shuffled_indices))

        # 0 1 2 3 4 5 6 7 8 9       -> 0 2 4 6 8
        # 0 1 2 3 4 5 6 7 8 9 10    -> 0 2 4 6 8 10 (Remove the last element for this case)

        # Remove the last element for this case
        if len(tokens) % 2 == 1:
            shuffled_indices.remove(max(shuffled_indices))

    elif type == ShuffleType.TRIGRAM:

        modified_indices = [idx for idx in original_indices if idx % 3 == 0]
        shuffled_indices = np.random.permutation(modified_indices)

        # Shufle the list of tokens until there is at least a change
        while all(shuffled_indices == modified_indices) and len(shuffled_indices) > 1:
            shuffled_indices = np.random.permutation(modified_indices)

        if strict:
            while any(shuffled_indices == modified_indices) and len(shuffled_indices) > 1:
                shuffled_indices = np.random.permutation(modified_indices)

        # Reconstruct pair-wise shuffled list
        shuffled_indices = [[idx, idx + 1, idx + 2] for idx in shuffled_indices]
        shuffled_indices = list(itertools.chain.from_iterable(shuffled_indices))

        # 0 1 2 3 4 5 6 7 8 9       -> 0 3 6 9 (Remove the last 2 elements for this case)
        # 0 1 2 3 4 5 6 7 8 9 10    -> 0 3 6 9 (Remove the last element for this case)
        # 0 1 2 3 4 5 6 7 8 9 10 11 -> 0 3 6 9

        # Remove the last 2 elements for this case
        if len(tokens) % 3 == 1:
            shuffled_indices.remove(max(shuffled_indices))
            shuffled_indices.remove(max(shuffled_indices))

        # Remove the last element for this case
        elif len(tokens) % 3 == 2:
            shuffled_indices.remove(max(shuffled_indices))

    elif type == ShuffleType.UNIGRAM:
        shuffled_indices = np.random.permutation(len(tokens))

        # Shufle the list of tokens until there is at least a change
        while all(shuffled_indices == np.arange(0, len(tokens))) and len(shuffled_indices) > 1:
            shuffled_indices = np.random.permutation(len(tokens))

        if strict:
            while any(shuffled_indices == np.arange(0, len(tokens))) and len(shuffled_indices) > 1:
                shuffled_indices = np.random.permutation(len(tokens))

    else:
        original_indices = np.arange(0, len(tokens))
        selected_idx = random.sample(original_indices, 1)
        original_indices.remove(selected_idx)

        shuffled_indices = copy.deepcopy(original_indices)
        shuffled_indices.insert(randrange(len(original_indices) + 1), selected_idx)

        while all(shuffled_indices == np.arange(0, len(tokens))) and len(shuffled_indices) > 1:
            shuffled_indices = copy.deepcopy(original_indices)
            shuffled_indices.insert(randrange(len(original_indices) + 1), selected_idx)

    shuffled_tokens = [tokens[idx] for idx in shuffled_indices]

    return shuffled_tokens, shuffled_indices
def shuffle_text(text, type=ShuffleType.UNIGRAM, strict=False, keep_punctuation=False):

    # ThangPM's NOTES 27-07-2020
    # Step 1: Split one sentence → [prefix][punctuations]
    # Step 2: Split [prefix] by spaces and shuffle
    # Step 3: Merge shuffled [prefix] with [punctuations]

    sentence_to_be_shuffled, punctuation_part = split_sentence_to_prefix_and_punctuation_part(text, keep_punctuation=keep_punctuation)

    tokens = sentence_to_be_shuffled.strip().split(" ")
    shuffled_tokens, shuffled_indices = shuffle_tokens(tokens, type=type, strict=strict)

    return " ".join(shuffled_tokens) + punctuation_part, shuffled_indices

def swap_tokens(tokens, ngrams=2):
    indices = np.arange(0, len(tokens))
    swapped_indices = random.sample(list(indices), ngrams)
    tokens[swapped_indices[0]], tokens[swapped_indices[1]] = tokens[swapped_indices[1]], tokens[swapped_indices[0]]

    return tokens, swapped_indices
def swap_text(text, ngrams=2, keep_punctuation=False):

    # ThangPM's NOTES 11-02-2020 (Nov 2nd)
    # Step 1: Split one sentence → [prefix][punctuations]
    # Step 2: Split [prefix] by spaces and swap
    # Step 3: Merge shuffled [prefix] with [punctuations]

    sentence_to_be_swapped, punctuation_part = split_sentence_to_prefix_and_punctuation_part(text, keep_punctuation=keep_punctuation)

    tokens = sentence_to_be_swapped.strip().split(" ")
    swapped_tokens, swapped_indices = swap_tokens(tokens, ngrams=ngrams)

    return " ".join(swapped_tokens) + punctuation_part, swapped_indices

def swap_two_nouns(text, print_log=True, keep_punctuation=False, seed=42):
    '''
    Statistics:
        1. How many examples in total?
        2. How many examples that are updated? (Positive vs negative)
            a. Swap 2 noun phrases if list NP >= 2
                (PropNs: ['Sue', 'Bill']; Nouns: ['a book'])
                Original:   Sue gave to Bill a book.
                Swapped:    Bill gave to Sue a book.
            b. Swap 2 nouns if list N >= 2
                (PropNs: ['Mary']; Nouns: ['the dean', 'a genuine linguist'])
                Original:   They represented seriously to the dean Mary as a genuine linguist.
                Swapped:    They represented seriously to a genuine linguist Mary as the dean.
            c. If list NP == list N == 1, swap N and NP
        3. How many examples that are kept unchanged?
    '''

    def merge_tokens_if_necessary(texts, text):
        '''
        :param texts: A list of tokens and phrases.
        :param text:  Original text.
        :return: A new list of merged tokens and phrases which is equivalent to splitting by a white space.
        '''
        tokens = text.split(" ")
        idx_token, idx_item = 0, 0

        print(text)

        while idx_token < len(tokens):
            if tokens[idx_token] != texts[idx_item]:
                # Merge 2 parts into 1 token (e.g., "I" + "'m" => "I'm")
                if len(tokens[idx_token]) > len(texts[idx_item]):
                    texts[idx_item] += texts[idx_item + 1]
                    del texts[idx_item + 1]
                # This part has multiple tokens (e.g., "The sailors")
                else:
                    tokens[idx_token] += " " + tokens[idx_token + 1]
                    del tokens[idx_token + 1]
                    idx_item -= 1 # Stay at the same index (+1 later)
                    idx_token -= 1

            idx_item += 1
            idx_token += 1

        assert " ".join(tokens) == " ".join(texts)

        return texts

    def extract_noun_from_phrase(phrase, tokens):
        noun_list = []
        suffix = ","

        for sub_text in phrase.split(" "):
            if sub_text in tokens or (sub_text + suffix) in tokens: # HOT FIX: (sub_text + ",") in tokens
                pDoc = nlp_spacy(sub_text)
                for token in pDoc:
                    if token.pos_ == "PROPN" or token.pos_ == "NOUN":
                        noun_list.append(sub_text + (suffix if (sub_text + suffix) in tokens else ""))

        # Randomly pick 1 from a list if there are more than 2 nouns
        # print(text)
        if len(noun_list) > 0:
            return random.sample(noun_list, 1)

        return noun_list

    np.random.seed(seed)
    swapped = 0

    try:
        sentence_to_be_swapped, punctuation_part = split_sentence_to_prefix_and_punctuation_part(text, keep_punctuation=keep_punctuation)
        doc = nlp_spacy(sentence_to_be_swapped)
    except:
        print("Cannot parse this sentence: " + text)
        return text, swapped, np.arange(0, len(text.split(" ")))

    # -----------------------------------------------------------------------
    # Analyze syntax
    # -----------------------------------------------------------------------
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    prop_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    verbs = [token.text for token in doc if token.pos_ == "VERB"]

    # Find named entities, phrases and concepts (Uncomment if necessary)
    # for entity in doc.ents:
    #     print(entity.text, entity.label_)
    # -----------------------------------------------------------------------
    texts = [t.text for t in nlp_spacy(sentence_to_be_swapped)]     # A list of tokens and phrases
    tokens = [item.split(" ") for item in texts]                    # A list of tokens ONLY
    tokens = [item for sublist in tokens for item in sublist]
    swapped_items = []

    # Preprocess noun_phrases, prop_nouns and nouns
    noun_phrases = [item for item in noun_phrases if len(extract_noun_from_phrase(item, tokens)) > 0]
    prop_nouns = [item for item in prop_nouns if len(extract_noun_from_phrase(item, tokens)) > 0]
    nouns = [item for item in nouns if len(extract_noun_from_phrase(item, tokens)) > 0]

    if len(prop_nouns) >= 2 and all(x in texts for x in prop_nouns):
        # Swap 2 prop nouns
        # ThangPM: Merge tokens for preserving the original indices => NOT USE THIS TIME, WILL COME BACK LATER
        # merged_texts = merge_tokens_if_necessary(texts, sentence_to_be_swapped)
        swapped_items = random.sample(prop_nouns, 2)
        swapped = 1
    elif len(nouns) >= 2 and all(x in texts for x in nouns):
        # Swap 2 nouns
        # merged_texts = merge_tokens_if_necessary(texts, sentence_to_be_swapped)
        swapped_items = random.sample(nouns, 2)
        swapped = 2
    elif len(noun_phrases) >= 2 and all(x in texts for x in noun_phrases):
        # Swap 2 noun phrases
        # merged_texts = merge_tokens_if_necessary(texts, sentence_to_be_swapped)
        swapped_items = random.sample(noun_phrases, 2)
        swapped = 3

    if swapped > 0:
        # print(tokens)
        # print("Noun phrases:", noun_phrases)
        # print("PropNs:", prop_nouns)
        # print("Nouns:", nouns)
        merged_tokens = merge_tokens_if_necessary(tokens, sentence_to_be_swapped)
        shuffled_indices = np.arange(0, len(merged_tokens))

        swapped_indices = [merged_tokens.index(extract_noun_from_phrase(item, merged_tokens)[-1]) for item in swapped_items] # Get item at index -1 because it's often the main Noun
        swapped_texts = copy.deepcopy(merged_tokens)

        swapped_texts[swapped_indices[0]], swapped_texts[swapped_indices[1]] = swapped_texts[swapped_indices[1]], swapped_texts[swapped_indices[0]]
        shuffled_indices[swapped_indices[0]], shuffled_indices[swapped_indices[1]] = shuffled_indices[swapped_indices[1]], shuffled_indices[swapped_indices[0]]

        if print_log:
            print("--------------------------------------------------------------------")
            # print(text)
            print(texts)
            print(swapped_texts)
            print("Noun phrases:", noun_phrases)
            print("PropNs:", prop_nouns)
            print("Nouns:", nouns)
            # print("Verbs:", verbs)

        return " ".join(swapped_texts) + punctuation_part, swapped, shuffled_indices

    return text, swapped, np.arange(0, len(text.split(" ")))

def split_sentence_to_prefix_and_punctuation_part(text, keep_punctuation):
    punctuation_part = ""
    text = text.strip()

    if keep_punctuation:
        for char in text[::-1]:
            if char not in string.punctuation and char != " ":
                break
            else:
                punctuation_part += char

        punctuation_part = punctuation_part[::-1]

    sentence_to_be_shuffled = text[:(len(text) - len(punctuation_part))]

    return sentence_to_be_shuffled, punctuation_part

def extract_noun_from_phrase(phrase, tokens):
    noun_list = []
    for sub_text in phrase.split(" "):
        if sub_text in tokens:
            pDoc = nlp_spacy(sub_text)
            for token in pDoc:
                if token.pos_ == "PROPN" or token.pos_ == "NOUN":
                    noun_list.append(sub_text)

    # Randomly pick 1 from a list if there are more than 2 nouns
    if len(noun_list) > 0:
        return random.sample(noun_list, 1)

    return noun_list
def has_two_nouns_or_more(text):

    swapped = 0

    try:
        sentence_to_be_swapped, punctuation_part = split_sentence_to_prefix_and_punctuation_part(text, keep_punctuation=True)
        doc = nlp_spacy(sentence_to_be_swapped)
    except:
        print("Cannot parse this sentence: " + text)
        return swapped > 0

    # -----------------------------------------------------------------------
    # Analyze syntax
    # -----------------------------------------------------------------------
    noun_phrases = [chunk.text for chunk in doc.noun_chunks]
    prop_nouns = [token.text for token in doc if token.pos_ == "PROPN"]
    nouns = [token.text for token in doc if token.pos_ == "NOUN"]
    # -----------------------------------------------------------------------
    texts = [t.text for t in nlp_spacy(sentence_to_be_swapped)]     # A list of tokens and phrases
    tokens = [item.split(" ") for item in texts]                    # A list of tokens ONLY
    tokens = [item for sublist in tokens for item in sublist]

    # Preprocess noun_phrases, prop_nouns and nouns
    noun_phrases = [item for item in noun_phrases if len(extract_noun_from_phrase(item, tokens)) > 0]
    prop_nouns = [item for item in prop_nouns if len(extract_noun_from_phrase(item, tokens)) > 0]
    nouns = [item for item in nouns if len(extract_noun_from_phrase(item, tokens)) > 0]

    if len(prop_nouns) >= 2 and all(x in texts for x in prop_nouns):
        # Swap 2 prop nouns
        # ThangPM: Merge tokens for preserving the original indices => NOT USE THIS TIME, WILL COME BACK LATER
        # merged_texts = merge_tokens_if_necessary(texts, sentence_to_be_swapped)
        swapped = 1
    elif len(nouns) >= 2 and all(x in texts for x in nouns):
        # Swap 2 nouns
        # merged_texts = merge_tokens_if_necessary(texts, sentence_to_be_swapped)
        swapped = 2
    elif len(noun_phrases) >= 2 and all(x in texts for x in noun_phrases):
        # Swap 2 noun phrases
        # merged_texts = merge_tokens_if_necessary(texts, sentence_to_be_swapped)
        swapped = 3

    return swapped > 0