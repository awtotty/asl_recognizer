import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences
from sklearn.model_selection import KFold

class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on BIC scores
        best_score = float('inf')
        best_model = None

        for num_of_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # Create GaussianHMM model
                model = self.base_model(num_of_components)
                # Calculate number of free parameters
                num_of_parameters = num_of_components*num_of_components + 2*num_of_components*self.lengths[0] - 1

                # Calculate the BIC score
                score = -2 * model.score(self.X, self.lengths) + num_of_parameters * math.log(self.lengths[0])

                # Minimize the BIC score
                if score <= best_score:
                    best_score = score
                    best_model = model
            except:
                pass

        return best_model

class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Implement model selection based on DIC scores
        best_score = float('-inf')
        best_model = None

        for num_of_components in range(self.min_n_components, self.max_n_components+1):
            try:
                # Create GaussianHMM model
                model = self.base_model(num_of_components)

                # Calculate sum of other word probabilities
                sum_other_probs = 0
                num_of_other_words = 0
                for word in self.words:
                    if word is not self.this_word:
                        X_other,lengths_other = self.hwords[word]
                        try:
                            sum_other_probs += model.score(X_other,lengths_other)
                            num_of_other_words += 1
                        except:
                            pass

                # Calculate the DIC score
                score = model.score(self.X, self.lengths) - 1/num_of_other_words * sum_other_probs

                # Maximize the DIC score
                if score >= best_score:
                    best_score = score
                    best_model = model
            except:
                pass


        return best_model


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        best_score = float('-inf')
        best_model = None
        split_method = KFold(n_splits=2)

        # If there are enough sequences to split:
        if len(self.sequences) > 1:
            for num_of_components in range(self.min_n_components, self.max_n_components+1):
                # folds and sum_of_logL used to find avg score over all splits
                folds = 0
                sum_of_logL = 0

                # Create models on all of the splits for this num_of_components
                for cv_train_idx, cv_test_idx in split_method.split(self.sequences):
                    # Split data and create new X,lengths tuples for hmmlearn
                    train_X,train_lengths = combine_sequences(cv_train_idx, self.sequences)
                    test_X,test_lengths = combine_sequences(cv_test_idx, self.sequences)

                    # Create/train model on train_X,train_lengths and score on test_X,test_lengths
                    try:
                        model = GaussianHMM(n_components=num_of_components, covariance_type="diag", n_iter=1000,
                        random_state=self.random_state, verbose=False).fit(train_X, train_lengths)
                        sum_of_logL += model.score(test_X,test_lengths)
                        folds += 1
                    except:
                        pass

                # Find avg score over all splits for this model
                avg_score = sum_of_logL/folds if folds is not 0 else float('-inf')

                # Maximize the score
                if avg_score >= best_score:
                    best_score = avg_score
                    best_model = model

        return best_model
