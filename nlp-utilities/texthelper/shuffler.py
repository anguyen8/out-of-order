import json
import string
import numpy as np

np.random.seed(42)

from identity_token import IdentityToken
from texthelper.occluder import *

from enum import Enum
import itertools
from random import randrange
import copy
import spacy

nlp_spacy = spacy.load("en_core_web_sm")
merge_nps = nlp_spacy.create_pipe("merge_noun_chunks")
nlp_spacy.add_pipe(merge_nps)

class ShuffleType(Enum):
    UNIGRAM = 1
    BIGRAM = 2
    TRIGRAM = 3
    ONE_WORD = 4 # Randomly pick one word and insert it to another place in a sentence.


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
        original_indices = list(np.arange(0, len(tokens)))
        selected_idx = random.sample(original_indices, 1)[0]
        original_indices.remove(selected_idx)

        shuffled_indices = copy.deepcopy(original_indices)
        shuffled_indices.insert(randrange(len(original_indices) + 1), selected_idx)

        while all(shuffled_indices == np.arange(0, len(tokens))) and len(shuffled_indices) > 1:
            shuffled_indices = copy.deepcopy(original_indices)
            shuffled_indices.insert(randrange(len(original_indices) + 1), selected_idx)

    shuffled_tokens = [tokens[idx] for idx in shuffled_indices]

    return shuffled_tokens
def shuffle_text(text, type=ShuffleType.UNIGRAM, strict=False, keep_punctuation=False, seed=42):

    # ThangPM's NOTES 27-07-2020
    # Step 1: Split one sentence → [prefix][punctuations]
    # Step 2: Split [prefix] by spaces and shuffle
    # Step 3: Merge shuffled [prefix] with [punctuations]

    np.random.seed(seed)

    sentence_to_be_shuffled, punctuation_part = split_sentence_to_prefix_and_punctuation_part(text, keep_punctuation=keep_punctuation)

    tokens = sentence_to_be_shuffled.strip().split(" ")
    shuffled_tokens = shuffle_tokens(tokens, type=type, strict=strict)

    return " ".join(shuffled_tokens) + punctuation_part

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
        for sub_text in phrase.split(" "):
            if sub_text in tokens:
                pDoc = nlp_spacy(sub_text)
                for token in pDoc:
                    if token.pos_ == "PROPN" or token.pos_ == "NOUN":
                        noun_list.append(sub_text)

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
        return text, swapped

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
        swapped_indices = [tokens.index(extract_noun_from_phrase(item, tokens)[-1]) for item in swapped_items] # Get item at index -1 because it's often the main Noun
        swapped_texts = copy.deepcopy(tokens)
        swapped_texts[swapped_indices[0]], swapped_texts[swapped_indices[1]] = swapped_texts[swapped_indices[1]], swapped_texts[swapped_indices[0]]

        if print_log:
            print("--------------------------------------------------------------------")
            # print(text)
            print(texts)
            print(swapped_texts)
            print("Noun phrases:", noun_phrases)
            print("PropNs:", prop_nouns)
            print("Nouns:", nouns)
            # print("Verbs:", verbs)

        return " ".join(swapped_texts) + punctuation_part, swapped

    return text, swapped

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


def shuffle_n_grams_in_entire_sentences(words, selected_mode):
    shuffled_words = []

    # Shuffle bigram, trigram or 4-gram in the entire sentences
    start_idx = end_idx = 0
    for i in range(int(len(words) / selected_mode)):
        start_idx = i * selected_mode
        end_idx = min(start_idx + selected_mode, len(words))
        shuffled_words.append(words[start_idx + 1: end_idx] + [words[start_idx]])

    start_idx = end_idx
    # Shuffle the remaining words and add them to the shuffled list
    if start_idx + 1 < len(words):
        shuffled_words.append(words[start_idx + 1: len(words)] + [words[start_idx]])
    # If there is one word left, add it to the shuffled list
    elif start_idx + 1 == len(words):
        shuffled_words.append([words[start_idx]])

    shuffled_words = [item for sublist in shuffled_words for item in sublist]

    return shuffled_words

def shuffle_n_grams_in_part_of_sentences(words, selected_mode):
    if len(words) < selected_mode:
        selected_mode = len(words)

    # Shuffle bigram, trigram or 4-gram in the entire sentences
    ngrams_idx = np.random.randint(0, int(len(words) / selected_mode))
    start_idx = ngrams_idx * selected_mode
    end_idx = min(start_idx + selected_mode, len(words))

    n_grams = words[start_idx : end_idx]
    shuffled_indices = np.random.permutation(len(n_grams))
    while any(shuffled_indices == np.arange(0, len(n_grams))):
        shuffled_indices = np.random.permutation(len(n_grams))
    shuffled_n_grams = [n_grams[idx] for idx in shuffled_indices]

    shuffled_words = words[0 : start_idx] + shuffled_n_grams + words[end_idx : len(words)]

    return shuffled_words

# Generate a shuffled sentence
def shuffle_sentence(sentence, shuffle_type="DN"):
    if sentence is None:
        return sentence

    words = sentence.split(" ")
    shuffled_words = words.copy()

    if len(shuffled_words) == 2:
        shuffled_words[0], shuffled_words[1] = shuffled_words[1], shuffled_words[0]
    elif shuffle_type == "DN":
        # random.shuffle(shuffled_words)
        shuffled_indices = np.random.permutation(len(shuffled_words))
        while any(shuffled_indices == np.arange(0, len(shuffled_words))):
            shuffled_indices = np.random.permutation(len(shuffled_words))
        shuffled_words = [shuffled_words[idx] for idx in shuffled_indices]
    else:
        choices = []

        if shuffle_type == "D2":
            choices = [2]                   # Shuffle bigram
        elif shuffle_type == "D3":
            choices = [3] #[2, 3]           # Shuffle either bigram or trigram
        elif shuffle_type == "D4":
            choices = [4] #[2, 3, 4]        # Shuffle either bigram or trigram or 4-gram
        elif shuffle_type == "D5":
            choices = [2, 3, 4, -1]         # Shuffle either bigram or trigram or 4-gram or remove randomly 1 word
        elif shuffle_type == "D6":
            choices = [2, 3, 4, -1, -2]     # Shuffle either bigram or trigram or 4-gram or remove randomly 1-2 words

        # selected_mode = random.choice(choices)
        selected_idx = np.random.randint(0, len(choices))
        selected_mode = choices[selected_idx]

        # Remove randomly 1-2 words
        if selected_mode < 0:
            return occlude_sentence(sentence, count_words_removed=abs(selected_mode))

        # Shuffle bigram, trigram or 4-gram in the entire sentences
        # shuffled_words = shuffle_n_grams_in_entire_sentences(words, selected_mode=selected_mode)
        shuffled_words = shuffle_n_grams_in_part_of_sentences(words, selected_mode=selected_mode)

    return " ".join(shuffled_words)


if __name__ == '__main__':
    a = "Today is a very beautiful day."
    b = "Today is a beautiful day."
    c = "Rust remover."

    # swap_two_nouns("Mary listens to the Grateful Dead, she gets depressed.")
    # swap_two_nouns("The company was also one of OpTic Gaming's main sponsors during the legendary organization's run to their first Call of Duty Championship back in 2017")
    # swap_two_nouns(" ".join(['How', 'can', 'I', 'expand', 'my', 'IQ?']), print_log=True)
    # swap_two_nouns("How long did Phillips manage the Apollo missions?", print_log=True)
    swap_two_nouns("the film’s performances are thrilling.", print_log=True)

    # a_shuffled_bigram = shuffle_text(a, ShuffleType.BIGRAM, strict=True, keep_punctuation=True)
    # a_shuffled_trigram = shuffle_text(a, ShuffleType.TRIGRAM, strict=True, keep_punctuation=True)
    #
    # b_shuffled_bigram = shuffle_text(b, ShuffleType.BIGRAM, strict=True, keep_punctuation=True)
    # b_shuffled_trigram = shuffle_text(b, ShuffleType.TRIGRAM, strict=True, keep_punctuation=True)
    #
    # c_shuffled_bigram = shuffle_text(c, ShuffleType.BIGRAM, strict=True, keep_punctuation=True)
    # c_shuffled_trigram = shuffle_text(c, ShuffleType.TRIGRAM, strict=True, keep_punctuation=True)

    # print(a_shuffled_bigram)
    # print(a_shuffled_trigram)
    # print(b_shuffled_bigram)
    # print(b_shuffled_trigram)

    '''
    np.random.seed(42)
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)

    np.random.seed(42)
    x = 1
    y = 2
    z = x + y + x*y + x*x
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)

    np.random.seed(42)
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    np.random.seed(42)
    x = 1
    y = 2
    z = x + y + x*y + x*x
    x = 1
    y = 2
    z = x + y + x*y + x*x
    x = 1
    y = 2
    z = x + y + x*y + x*x
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    np.random.seed(42)
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    np.random.seed(42)
    x = 1
    y = 2
    z = x + y + x*y + x*x
    x = 1
    y = 2
    z = x + y + x*y + x*x
    a_shuffled_bigram = shuffle_text(a, ShuffleType.UNIGRAM, strict=False, keep_punctuation=True)
    '''




