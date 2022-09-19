import random
import numpy as np

random.seed(42)
np.random.seed(42)


# Randomly remove a word in a sentence
def occlude_sentence(sentence, count_words_removed=None):
    if sentence is None:
        return sentence

    words = sentence.split(" ")

    if len(words) == 1:
        return words[0]

    if count_words_removed is None:
        # count_words_removed = random.randrange(1, len(words))
        count_words_removed = np.random.randint(1, len(words))
    elif count_words_removed < 1:
        count_words_removed = (int)(count_words_removed * len(words))

    for i in range(count_words_removed):
        # removed_index = random.randrange(0, len(words))
        removed_index = np.random.randint(0, len(words))
        del words[removed_index]

    return " ".join(words)