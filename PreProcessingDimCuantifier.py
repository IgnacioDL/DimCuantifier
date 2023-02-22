import gensim
import nltk

from gensim.scripts.glove2word2vec import glove2word2vec

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
from numpy import linalg
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string


class PreProcessingDimCuantifier():
    """ A class to help with the preprocessing that can be needed when using DimCuantifier class.
    Some of this functions are based on the code of the Polar framework paper from its github 
    repository: https://github.com/Sandipan99/POLAR


    Attributes:
    
        stop_words_and_symbols (list) : 
            List of stopwords and symbols to be removed from the text


    """

    def __init__(self):
        # Set stopwords from nltk and punctuation symbols from string
        self.stop_words_and_symbols = set(stopwords.words('English') + list(string.punctuation))


    def generate_normalized_embeddings(self, model, output_path):
        """ Generate a file with the normalized embeddings of the words in the model

        Args:
            model (gensim.model): Word embedding model to obtain vector representation for words
            output_path (str): Path to save the file with the normalized embeddings
        """

        # Open file to write
        temp_file = open(output_path,'wb')

        # Write the number of words and the dimension of the embeddings
        temp_file.write(str.encode(str(len(model.key_to_index))+' '+str(model.vector_size)+'\n'))
        
        for each_word in model.key_to_index:
            temp_file.write(str.encode(each_word+' '))
            temp_file.write(model[each_word]/linalg.norm(model[each_word]))
            temp_file.write(str.encode('\n'))
        
        temp_file.close()
    

    def change_stopwords_language(self, language):
        """ Change the language of the stopwords and symbols to be removed from the text

        Args:
            language (str): Language to be used. It must be a language supported by nltk.corpus.stopwords
        """        

        self.stop_words_and_symbols = set(stopwords.words(language) + list(string.punctuation))
    

    def list_polar_words_tuple(self, file_list, model):
        """ Reed file of words formated as: lower-n	raise-n	False	antonym
        This format can be found in the examples in the repository

        Args:
            file_list (str): Path of the file of list of words
            model (gensim.model): Word embedding model to obtain vector representation for words

        Returns:
            list(tuple(str)): List of tuples with the pair of antomym words
        """

        list_antonym = []

        # Go through each file
        for file in file_list:
            # Open file
            with open(file) as fp:
                # Go through each line of the file
                for line in fp:
                    # Split the line
                    parts = line.split()
                    # Check if the line is an antonym
                    if parts[3]=='antonym':
                        # Get the words and check if they are in the model
                        word1 = parts[0].split('-')[0]
                        word2 = parts[1].split('-')[0]
                        if word1 in model and word2 in model:
                            list_antonym.append((word1.strip().lower(), word2.strip().lower()))

        # return list_antonym
        return list(dict.fromkeys(list_antonym).keys())


    def select_polar_words_list(self, model, list_polar_word):
        """ Given a list of polar words, select the pair with less value for cosine similiraty
        between the two words

        Args:
            model (gensim.model): Word embedding model to obtain vector representation for words
            list_polar_word (list(tuple(str))): List of tuples with the pair of antomym words

        Returns:
            list(tuple(str)): List of tuples with the pair of antomym words selected to minimize
                the cosine similarity
        """        

        # Add the value of cosine similarity between two polar words
        all_similarity = defaultdict(dict)
        for each_pair in list_polar_word:
            word1 = min(each_pair[0], each_pair[1])
            word2 = max(each_pair[0], each_pair[1])
            # Add the value of cosine similarity between two polar words
            all_similarity[word1][word2] = abs(cosine_similarity([model[word1]],[model[word2]])[0][0])

        # Select the less value for cosine similiraty between the two words
        final_polar_word_list = []
        for index_counter, each_key in enumerate(all_similarity):
            # Sort the list of polar words by the value of cosine similarity
            listofTuples = sorted(all_similarity[each_key].items() ,  key=lambda x: x[1])
            # Add the pair of polar words with less value for cosine similiraty
            final_polar_word_list.append((each_key, listofTuples[0][0]))
        
        return final_polar_word_list

    
    def preprocess_document(self, document, model):
        """ Preprocess a document to be used with DimCuantifier class

        Args:
            document (str): content of the document in a string
            model (gensim.model): Word embedding model to obtain vector representation for words

        Returns:
            list: List of the tokenized content of the document. Stopwords and symbols are removed
                and only words in the model are kept
        """        

        # Tokenize the document
        tokenized_document = word_tokenize(document.lower())

        # Remove stopwords and symbols and keep only words in the model
        filtered_sentence = [word for word in word_tokenize(document.lower()) 
                                  if not word in self.stop_words_and_symbols and word in model.index_to_key]

        return filtered_sentence