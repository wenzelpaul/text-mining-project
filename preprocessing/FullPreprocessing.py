"""
This module provides all steps that are necessary to pre-process the DRI corpus for the machine learning techniques.
"""

import zipfile
import xml.etree.ElementTree
import operator
import re

import Config
from preprocessing.PreprocessSentences import process_all_layers


class Layer:
    """
    This subclass provides an object structure for the layers of the DRI corpus to simplify the preprocessing execution.

    Args:
        class_name (string):        The class name.
        class_description (string): The class description.
        element_name (string):      The element name.
        tag_name (string):          The tag name.

    Attributes:
        class_name (string):        The class name.
        class_description (string): The class description.
        element_name (string):      The element name.
        tag_name (string):          The tag name.
    """

    def __init__(self, class_name, class_description, element_name, tag_name):
        self._name = class_name
        self._description = class_description
        self._element = element_name
        self._tag = tag_name

    def addSublayer(self, sub_element_name, _sub_tag_prefix):
        self._sub_element = sub_element_name
        self._sub_tag = _sub_tag_prefix


class FullPreprocessing:
    """
    Performs all preprocessing steps: corpus reading, relevant data extraction, refinement of data, mapping of
    sentences to feature vectors, and storing all relevant data to dedicated csv-files.

    Attributes:
        layers (list of 4 Layer objects): list of all 4 layers to simplify the preprocessing execution.
    """

    def __init__(self):

        # create layers
        self._layers = [
            Layer('RHETORICAL',
                  'Scientific Discourse',
                  'Sentence',
                  'rhetoricalClass'),
            Layer('ASPECT',
                  'Subjective Statements',
                  'Sentence',
                  'aspectClass'),
            Layer('CITATION_PURPOSE',
                  'Only for in-line citations',
                  'InlineCitation',
                  'CITid'),
            Layer('SUMMARY',
                  'Summary relevance specified in grades from 1.0 (totally irrelevant) to 5.0 (very relevant)',
                  'Sentence',
                  'summaryRelevanceScore')
        ]
        # add sublayer for citation layer
        self._layers[2].addSublayer('Cit_context', 'CITid_')

    def execute(self):
        """
        Executes all preprocessing steps.
        """

        # read zipped DRI corpus
        dri_corpus = zipfile.ZipFile(Config.PATH_DRI_CORPUS)
        filenames = [file_info.filename for file_info in dri_corpus.filelist]

        # create vectors with respectively 4 values for the 4 layers
        # (-> simplifies access within layer loop)

        # init lists of each category to pass to preprocessing
        sentence_vec = [[], [], [], []]
        # init lists for unique annotation values as strings
        label_list_vec = [[], [], [], []]
        # init dictionaries for mapping each annotation value with a label of a unique integer
        label_dict_vec = [dict(), dict(), dict(), dict()]
        # init vectors for output files
        label_outputfile_vec = []
        dict_int_label_outputfile_vec = []
        # open output files and store references into the vectors above
        for i in range(0, len(self._layers)):
            label_outputfile_vec.append(open(Config.OUTPUT_PATHS[i] + Config.OUTPUT_FILETYPES[2], 'w', encoding="utf8"))
            dict_int_label_outputfile_vec.append(open(Config.OUTPUT_PATHS[i] + Config.OUTPUT_FILETYPES[3], 'w'))

        # iterate all four layers (0:rhetorical, 1:aspect, 2:citation purpose, 3:summary)
        for index, layer in enumerate(self._layers):

            csv_string_body = ''

            # iterate xml files (leave out summary .txt files)
            for filename in [filename for filename in filenames if layer._name in filename]:

                # read current file (out of total 4*40 files)
                content = dri_corpus.read(filename)

                # remove CitSpans by removing REGEX in string
                content = re.sub(' <CitSpan>[^>]*</CitSpan>', '', str(content, 'utf-8'))

                # get xml root tree from string
                root = xml.etree.ElementTree.fromstring(content)

                # get list of all citation ids within document (needed for Cit_context value in CITid_? tag)
                if index == 2:
                    citation_ids_list = []
                    for sentence in root.findall('InlineCitation'):
                        id = str(sentence.get(layer._tag))
                        if not citation_ids_list.__contains__(id):
                            citation_ids_list.append(id)

                # search Cit_context in layer 3, in layer 1,2 and 4 for Sentence
                search_tag = 'Cit_context' if index == 2 else 'Sentence'

                for sentence in root.findall(search_tag):
                    # get sentence content as string
                    sentence_content = str(sentence.text)

                    # remove newlines occurring within a sentence
                    sentence_content = sentence_content.replace(Config.CSV_CHAR_NEW_ROW, '')

                    # skip empty sentences
                    if sentence_content is '': continue

                    # skip sentences with less than a number of Config.MIN_NUM_WORDS_IN_SENTENCE words
                    if (len(sentence_content.split()) < Config.MIN_NUM_WORDS_IN_SENTENCE): continue

                    # add annotation label if it has not occurred yet
                    label = str(sentence.get(layer._tag))

                    # map citation id within id list (determine ? of CITid_?)
                    if search_tag == 'Cit_context':
                        for id in citation_ids_list:
                            label_cit = str(sentence.get('CITid_' + id))
                            if label_cit != 'None':
                                label = label_cit

                    # skip following classes as they have too low instances
                    if (label == 'DRI_Challenge_Hypothesis' or label == 'SUBSTANTIATION'): continue

                    # add current sentence string to list for feature vector generation
                    sentence_vec[index].append(sentence_content)

                    # add to label list and dict if label has not occurred yet
                    if not label_list_vec[index].__contains__(label):
                        label_list_vec[index].append(label)
                        label_dict_vec[index][label] = len(label_dict_vec[index])
                    # write annotation value to output
                    csv_string_body += str(label_dict_vec[index][label]) + Config.CSV_CHAR_NEW_ROW

            label_dict_sorted_by_int = sorted(label_dict_vec[index].items(), key=operator.itemgetter(1))

            csv_string_head = ''
            for i, t in enumerate(label_dict_sorted_by_int):
                csv_string_head += str(t[0])
                if i < len(label_dict_sorted_by_int) - 1:  # leave out separation in last value
                    csv_string_head += Config.CSV_CHAR_NEW_COLUMN

            dict_int_label_outputfile_vec[index].write(csv_string_head)
            label_outputfile_vec[index].write(csv_string_body)

        # execute further preprocessing steps based on string sentences
        process_all_layers(sentence_vec)

        # close files
        for i in range(0, len(self._layers)):
            label_outputfile_vec[i].close()
            dict_int_label_outputfile_vec[i].close()
