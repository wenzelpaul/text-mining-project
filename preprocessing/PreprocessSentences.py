"""
This module creates the feature vectors and saves them in csv-format.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import Config

nlp = spacy.load('en')
adj_adv_list = []
verb_list = []
noun_list = []


class LemmaTokenizer(object):
    """
    This class uses spacy's natural language processing to tokenize, lemmanize and do part of speech tagging.
    """

    def __call__(self, doc):
        """
        Tokenize a sentence and lemmanize each token. Count occurrences of ADV, ADJ, VERB and NOUN in each sentence.

        Args:
            doc (str):        The sentence to process.

        Returns: list
            Return the list of lemmanized tokens.
        """
        lemmanized_tokens = []
        pos_tokens = []
        for word in nlp(doc):
            if len(
                    word.text) > 2 and word.lemma_ != '-PRON-':  # filter punctuation, only tokens with at least 3 characters
                lemmanized_tokens.append(word.lemma_)
                pos_tokens.append(word.pos_)
        if (len(pos_tokens) > 0):
            adj_adv_list.append((pos_tokens.count("ADV") + pos_tokens.count("ADJ")) / len(pos_tokens))
            verb_list.append(pos_tokens.count("VERB") / len(pos_tokens))
            noun_list.append(pos_tokens.count("NOUN") / len(pos_tokens))
        else:  # in case of an empty sentence
            adj_adv_list.append(0)
            verb_list.append(0)
            noun_list.append(0)
        return lemmanized_tokens


def init_vectorizer(range, max_features):
    """
    Create and configure a TfidfVectorizer.

    Args:
        range (int):        Set to uni-, bi- or tri-gram.
        max_features (int): The maximum amount of features.

    Returns: TfidfVectorizer
        Return the vectorizer.
    """
    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(),
                                 ngram_range=(range, range),
                                 stop_words=Config.STOP_WORD_LANGUAGE,
                                 max_features=max_features
                                 )
    return vectorizer


unigram_vectorizer = init_vectorizer(1, Config.MAX_FEATURES_UNIGRAM)
bigram_vectorizer = init_vectorizer(2, Config.MAX_FEATURES_BIGRAM)
trigram_vectorizer = init_vectorizer(3, Config.MAX_FEATURES_TRIGRAM)


def process_all_layers(sentence_vec):
    """
    Start the preprocessing for each layer with the sentences provided by sentence_vec.

    Args:
        sentence_vec (array):   An array containing the four layers with the sentences of each layer.
    """
    for i in range(0, len(sentence_vec)):
        _process_sentence(sentence_vec[i], Config.OUTPUT_PATHS[i])


def _process_sentence(sentence_list, path):
    """
    This method does the preprocessing of one layer.

    Args:
        sentence_list (list):   The sentences to process
        path (str):             The path to save the feature vector.
    """
    preprocessed_unigram_features = unigram_vectorizer.fit_transform(sentence_list)
    preprocessed_bigram_features = bigram_vectorizer.fit_transform(sentence_list)
    preprocessed_trigram_features = trigram_vectorizer.fit_transform(sentence_list)

    vec_matrix_unigram = preprocessed_unigram_features.toarray()
    vec_matrix_bigram = preprocessed_bigram_features.toarray()
    vec_matrix_trigram = preprocessed_trigram_features.toarray()

    rows_unigram, cols_unigram = preprocessed_unigram_features.shape
    rows_bigram, cols_bigram = preprocessed_bigram_features.shape
    rows_trigram, cols_trigram = preprocessed_trigram_features.shape

    outputfile_feature_vector_words = open(path + Config.OUTPUT_FILETYPES[1], 'w')
    outputfile_feature_vector = open(path + Config.OUTPUT_FILETYPES[0], 'w')

    first_row = "ADJ-ADV;VERB;NOUN;"
    unigram_feature_names = ';'.join(unigram_vectorizer.get_feature_names())
    bigram_feature_names = ';'.join(bigram_vectorizer.get_feature_names())
    trigram_feature_names = ';'.join(trigram_vectorizer.get_feature_names())

    outputfile_feature_vector_words.write(
        first_row + unigram_feature_names + ';' + bigram_feature_names + ';' + trigram_feature_names)

    i = 0
    for row in range(rows_unigram):
        adv_verb_noun = str(adj_adv_list[i]) + ';' + str(verb_list[i]) + ';' + str(noun_list[i]) + ';'

        unigram_features = ';'.join([str(vec_matrix_unigram[row][column]) for column in range(cols_unigram)])
        bigram_features = ';'.join([str(vec_matrix_bigram[row][column]) for column in range(cols_bigram)])
        trigram_features = ';'.join([str(vec_matrix_trigram[row][column]) for column in range(cols_trigram)])

        outputfile_feature_vector.write(
            adv_verb_noun + unigram_features + ';' + bigram_features + ';' + trigram_features + '\n')
        i = i + 1
    outputfile_feature_vector.close()
    outputfile_feature_vector_words.close()
