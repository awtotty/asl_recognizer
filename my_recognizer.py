import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    probabilities = []
    guesses = []
    # Implement the recognizer
    # Iterate through the test_set unknown words
    for word_id in range(len(test_set.get_all_sequences())):
            # Create a dictionary for this test word with keys = known words and values = score from model applied to test word
            prob_dict = dict()
            test_X,test_lengths = test_set.get_all_Xlengths()[word_id]
            for known_word,model in models.items():
                try:
                    prob_dict[known_word] = model.score(test_X,test_lengths)
                except:
                    pass
            # Add this dictionary to probabilities
            probabilities.append(prob_dict)
            # Add the known_word with highest prob to guesses.
            best_word = max(prob_dict, key=prob_dict.get)
            guesses.append(best_word)
    # return probabilities, guesses
    return probabilities, guesses
