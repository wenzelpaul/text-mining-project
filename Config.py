"""
This module stores relevant configuration parameters that are accessed from the other modules.
"""

# Paths to the zipped DRI Corpus, output folder for preprocessed data and evaluation results
PATH_DRI_CORPUS     = '../dri_corpus.zip'
PREPROCESSING_DIR   = '../data_preprocessed/'
EVALUATION_DIR      = '../data_evaluation/'

############################################################
# Pre-processing
############################################################

STOP_WORD_LANGUAGE          = 'english'
MAX_FEATURES_UNIGRAM        = 300
MAX_FEATURES_BIGRAM         = 75
MAX_FEATURES_TRIGRAM        = 25
MIN_NUM_WORDS_IN_SENTENCE   = 4 # sentences less than this number will be ignored

# csv separation characters for new line/row and new column
CSV_CHAR_NEW_ROW    = '\n'
CSV_CHAR_NEW_COLUMN = ';'

# output paths and names for the csv files that store all preprocessed data
OUTPUT_PATHS = [
    PREPROCESSING_DIR + 'rhetorical/',
    PREPROCESSING_DIR + 'aspect/',
    PREPROCESSING_DIR + 'citation_purpose/',
    PREPROCESSING_DIR + 'summary/'
]
OUTPUT_FILETYPES = [
    'data.csv',
    'feature_names.csv',
    'target.csv',
    'target_names.csv'
]

############################################################
# ML algorithm evaluation
############################################################

RANDOMIZED_SEARCH_ITERATIONS    = 15
CROSS_VALIDATION_FOLDS          = 10

# parameters for regression
CV_FOLDS_REGRESSION             = 5
TRAIN_SET_START                 = 0
TRAIN_SET_END                   = 9971
TEST_SET_STOP                   = 9000
TEST_SET_SIZE                   = 1000