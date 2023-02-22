import gensim
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial import distance
from numpy import linalg
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import preprocessing
from tabulate import tabulate


class DimCuantifier:
    """
    A class used to cuantify complexity or number of dimensions needed to represent certain
    variance percentage value in a text. That can be done using polar embeddings or word embeddings.
    This class also implements methods described in Polar, SemAxis and FrameAxis frameworks.


    Attributes:
    
        model (gensim.model) : 
            Word embedding model to obtain vector representation for words

        polar_pairs_list (list(tuple(str))) : 
            List of polar word pairs to use for the semantic axis

        semantic_axis_vector_list (list(np.array)) : 
            List of calculated semantic axis (substraction between vectors of a polar word pair)
        
        corpus (list(list(str))) : 
            Collection of documents. Every document is a list of words
        
        corpus_len (int) : 
            Number of documents in the corpus
        
        word_frequency_dictionary (dict(str, int)) : 
            Dictionary with word frequency for every word in corpus
        
        n_arbitrary_dimensions (int) : 
            Number of arbitrary dimensions to use to select polar dimensions when not all are utilized
        
        bias_list (list(float)) : 
            List of bias values of the corpus for every semantic axis
        
        intensity_list (list(float)) : 
            List of intensity values of the corpus for every semantic axis
        
        bias_intensity_df (pd.DataFrame) : 
            Bias and Intensity of the entire corpus for every semantic axis
        
        percentage_var (float) : 
            Percentage of variance to be represented in PCA
        
        dimensions_index (list(int)) : 
            When selecting dimensions this list contains the index of the dimensions selected
        
        dir_matrix (list(np.array)) : 
            Direction matrix or the result of stacking polar vectors
        
        dir_matrix_transpose_inverted (list(np.array)) : 
            Inverted transpose of the direction matrix
        
        word_polar_embeddings (dict(str, np.array)) : 
            Polar embeddings of every word in the corpus
        
        document_polar_embeddings (list(np.array)) : 
            Polar embeddings of every document in the corpus
        
        corpus_polar_embedding (np.array) : 
            Polar embeddings of the corpus
        
        n_dimensions_corpus (int) : 
            Complexity or number of dimensions of the corpus that explain the percentage of
            variance using polar embeddings
        
        n_dimensions_documents_list (list(int)) : 
            Complexity or number of dimensions of every document in the corpus that explain
            the percentage of variance using polar embeddings
        
        pca_corpus (PCA) : 
            PCA object for the corpus using polar embeddings
        
        pca_documents_list (list(PCA)) : 
            PCA objectd for every document in the corpus using polar embeddings

        loading_scores_corpus (pd.DataFrame)) : 
            Dataframe with the loading scores of the corpus using polar embeddings
        
        loading_scores_documents (list(pd.DataFrame)) : 
            List of Dataframes with the loading scores of every document using polar embeddings
        
        n_dimensions_corpus_WE (int) : 
            Complexity or number of dimensions of the corpus that explain the percentage of
            variance using word embeddings
        
        n_dimensions_documents_list_WE (list(int)) : 
            Complexity or number of dimensions of every document in the corpus that explain
            the percentage of variance using word embeddings
        
        pca_corpus_WE (PCA) : 
            PCA object for the corpus using word embeddings
        
        pca_documents_list_WE (list(PCA)) : 
            PCA objectd for every document in the corpus using word embeddings
        

    """

    def __init__(self, model, polar_pairs_list):
        self.model = model
        self.polar_pairs_list = polar_pairs_list

        self.semantic_axis_vector_list = []
        self.corpus = []
        self.corpus_len = 0
        self.word_frequency_dictionary = Counter()

        self.n_arbitrary_dimensions = 10
        self.bias_list = []
        self.intensity_list = []
        self.bias_intensity_df = None
        
        self.percentage_var = 0.99
        self.dimensions_index = []
        self.dir_matrix = []
        self.dir_matrix_transpose_inverted = []
        self.word_polar_embeddings = defaultdict()
        self.document_polar_embeddings = []
        self.corpus_polar_embedding = None
        self.n_dimensions_corpus = None
        self.n_dimensions_documents_list = []
        self.pca_corpus = None
        self.pca_documents_list = []
        self.loading_scores_corpus = None
        self.loading_scores_documents = []

        self.n_dimensions_corpus_WE = None
        self.n_dimensions_documents_list_WE = []
        self.pca_corpus_WE = None
        self.pca_documents_list_WE = []

        self.__generate_semantic_axis_vector_list()


    def __generate_semantic_axis_vector_list(self):
        """ This function generates a list with semantic axis vector for every polar pair
        given in polar_pairs_list and save it in semantic_axis_vector_list as
        a numpy array in the same order
        """

        # Iterate polar pairs list
        for each_word_pair in self.polar_pairs_list:

            # Append the substraction between the two word embedding representation
            # the substraction is called semantic axis
            self.semantic_axis_vector_list.append(
                self.model[each_word_pair[0]] - self.model[each_word_pair[1]])

        # Transform the semantic axis vector list to a numpy array
        self.semantic_axis_vector_list = np.array(
            self.semantic_axis_vector_list)


    def __calculate_word_frequency_dictionary(self):
        """ This function calculates word frequency for every word in corpus and add
        it to a dictionary word_frequency_dictionary. It is immediately called when 
        a corpus is added. It also saves the lenght of the corpus vocabulary
        """

        # Check if there is a corpus set
        if not self.corpus:
            print("Set corpus first")  ## TO DO: Exception
            return

        # Clear current word frequency dictionary to calculate again
        self.word_frequency_dictionary.clear()

        # Go over every document in corpus and add it to the dictionary
        for document in self.corpus:
            self.word_frequency_dictionary.update(document)
        
        # Save total number of words
        self.corpus_len = sum(self.word_frequency_dictionary.values())

    
    def __calculate_dir_matrix_from_all_semantic_axis(self):
        """ The direction matrix is calculated from semantic_axis_vector_list
        to create polar embedding subspace. This functionality is separated of
        __invert_dir_matrix because there is cases when only some semantic axis
        are selected to create direction matrix (see select_dim_by_intensity)
        """

        # Start direction matrix with the semantic axis vectors
        self.dir_matrix = self.semantic_axis_vector_list

        # Invert and transpose to create polar embedding subspace
        self.__invert_dir_matrix()


    def __invert_dir_matrix(self):
        """ This function transpose, invert and save the direction matrix in 
        dir_matrix_transpose_inverted. The direction matrix needs to be already set.
        This operation needs to be done to find the polar embedding subspace as stated
        in the Polar framework paper. Having dir_matrix and dir_matrix_transpose_inverted
        as separate variables is to keep the structure of Polar framework
        """
        
        # Check if dir matrix is set
        if self.dir_matrix == []:
            print('dir matrix needs to be set')  ## TO DO: Exception

        self.dir_matrix_transpose_inverted = np.linalg.pinv(
            np.transpose(self.dir_matrix))


    def set_n_arbitrary_dimensions(self, k):
        """ Set n arbitrary dimensions to select according to some criteria. This number will
        only be used to select an arbitrary amount of dimensions as it is done in the 
        Polar framework paper

        Args:
            k (int): number of dimensions
        """

        # Set number of arbitrary dimensions
        self.n_arbitrary_dimensions = k


    def set_percentage_var(self, percentage):
        """ This function sets the percentage of variance to be represented in PCA

        Args:
            percentage (float): Percentage of variance
        """

        self.percentage_var = percentage


    def set_corpus(self, corpus):
        """ This functions sets a corpus to work with. Besides, it calls the function to
        calculate word frequency of the corpus

        Args:
            corpus (list(list(string))): It is a list of documents. Documents are also
                lists of tokenized strings that represents a text
        """

        # Set corpus
        self.corpus = corpus
        
        # Calculate word frequency of the corpus
        self.__calculate_word_frequency_dictionary()


    def add_to_corpus(self, corpus):
        """ This function concatenates a new corpus to an already set corpus. Besides, 
        it calculates word frequency of the new corpus

        Args:
            corpus (list(list(string))): It is a list of documents. Documents are also
                lists of tokenized strings that represents a text
        """

        # Add new corpus to the previos one set
        self.corpus = self.corpus + corpus
        
        # Calculate word frequency of the new corpus
        self.__calculate_word_frequency_dictionary()
    

    def get_polar_pairs_list(self):
        """ Returns the polar pair list

        Returns:
            list(tuple(str)): List of polar pairs
        """

        return self.polar_pairs_list
    

    def get_semantic_axis_vector_list(self):
        """ Returns a list with vectors that represents semantic axis

        Returns:
            list(list(float)): List of semantic axis vectors
        """
        
        return self.semantic_axis_vector_list


    def get_corpus(self):
        """ Returns corpus

        Returns:
            list(list(string)): It is a list of documents. Documents are also
                lists of tokenized strings that represents a text
        """

        return self.corpus


    def get_word_frequency_dictionary(self):
        """ Returns word frequency dictionary of the current corpus

        Returns:
            dict: Dictionary of all words that appeared in the corpus with its frequency as value
        """

        return self.word_frequency_dictionary


    def get_n_polar_pairs(self):
        """ Returns number of polar pairs given

        Returns:
            int: length of polar_pairs_list
        """

        # Return number of polar pairs
        return len(self.polar_pairs_list)
    

    def get_word_polar_embeddings(self):
        """ Returns dictionary of polar embeddings for words

        Returns:
            dict[str]: Dictionary of words with its polar embedding as value
        """

        return self.word_polar_embeddings
    

    def get_document_polar_embeddings(self):
        """ Returns dictionary of polar embeddings for documents

        Returns:
            dict[str]: Dictionary of documents with its polar embedding as value
        """

        return self.document_polar_embeddings
    

    def get_corpus_polar_embedding(self):
        """ Returns the corpus polar embedding

        Returns:
            np.array: Numpy array of the corpus polar embedding
        """

        return self.corpus_polar_embedding
    

    def get_loading_scores(self, on='corpus'):
        """ Return loading scores for the corpus or documents. The only principal components that 
        appear in this DataFrame are the ones that have the variance needed to reach the percentage of 
        variance set

        Returns:
            pd.DataFrame: Pandas DataFrame of loading scores of the principal components of the 
                          corpus or documents
        """

        if on == 'corpus':
            return self.loading_scores_corpus
        elif on == 'documents':
            return self.loading_scores_documents
        else:
            print('on must be either corpus or documents')


    def get_percentage_var(self):
        """ Return the percentage of dimensions sets

        Returns:
            float: Percentage of dimensions
        """

        return self.percentage_var


    def contribution(self, word_vector, semantic_axis_vector):
        """ This functions calculates contribution of a word_vector to a semantic
        axis vector, calculating cosine similarity between both. It is named contribution
        as it is in FrameAxis framework paper

        Args:
            word_vector (float): Vector that represents a word
            semantic_axis_vector (float): Vector that represents a semantic axis

        Returns:
            float: Contribution or cosine similarity of a word to a semantic axis
        """

        # Return cosine similarity between word and semantic axis
        return 1.0 - distance.cosine(word_vector, semantic_axis_vector)


    def bias_corpus_for_semantic_axis(self, semantic_axis_vector):
        """ This functions calculates bias for a corpus given a semantic axis vector.
        Bias is a metric proposed by the FrameAxis framework paper and is calculated as
        the sum of contribution for every word to a semantic axis vector, multiplied for the
        frecuency of the word. The completed sum is divided for the amount of words in corpus

        Args:
            semantic_axis_vector (list(float)): Vector that represents a semantic axis

        Returns:
            float: Total bias of the corpus for a semantic axis
        """

        # Check if there is a corpus set
        if not self.corpus:
            print("Set corpus first")  ## TO DO: Exception
            return

        total_bias = 0.0

        # Go over every word in the word frequency dictionary
        for word in self.word_frequency_dictionary:
            
            # Add the bias for the word to the total bias
            # Bias for the word is calculated as the contribution of the word to the 
            # semantic axis, multiplied by the frequency of the word in the corpus
            total_bias += self.contribution(self.model[word],
                                            semantic_axis_vector) * self.word_frequency_dictionary[word]
        
        # Divide total bias in corpus length
        total_bias /= self.corpus_len

        return total_bias


    def intensity_corpus_for_semantic_axis(self, semantic_axis_vector, corpus_bias=None):
        """ This functions calculates intensity for a corpus given a semantic axis vector.
        Intensity is a metric proposed by the FrameAxis framework paper and is calculated as 
        the sum of contribution of a word to a semantic axis vector minus total bias of the 
        corpus, squared and multiplied for the frecuency of the word. The completed sum is 
        divided for the amount of words in corpus

        Args:
            semantic_axis_vector (list(float)): Vector that represents a semantic axis
            corpus_bias (float, optional): Total bias of the corpus for a semantic axis, if it is not
                given, it is calculated. Defaults to None.

        Returns:
            float: Total intensity of the corpus for a semantic axis
        """

        # Check if there is a corpus set
        if not self.corpus:
            print("Set corpus first")  ## TO DO: Exception
            return

        # Check if corpus bias is set, if not, calculate it
        if not corpus_bias:
            corpus_bias = self.bias_corpus_for_semantic_axis(
                semantic_axis_vector)

        total_intensity = 0.0

        # Go over every word in the word frequency dictionary
        for word in self.word_frequency_dictionary:

            # Add the intensity for the word to the total intensity
            # Intensity for the word is calculated as the contribution of the word to the 
            # semantic axis minus the total corpus bias, squared and then multiplied by the 
            # frequency of the word in the corpus 
            total_intensity += ((self.contribution(
                self.model[word], semantic_axis_vector) - corpus_bias) ** 2) * self.word_frequency_dictionary[word]

        # Divide total intensity in corpus length
        total_intensity /= self.corpus_len

        return total_intensity


    def calculate_bias_and_intensity(self):
        """ This functions calculates Bias and Intensity of the entire corpus 
        for every semantic axis in semantic_axis_vector_list. It saves a 
        pandas DataFrame storing bias and intensity values for each semantic
        axis sorted by intensity values descreasingly

        Returns:
            pd.DataFrame: Returns pandas dataframe with every polar pair and its
                bias and intensity in the corpus
        """

        self.bias_list = []
        self.intensity_list = []

        # Go over every semantic axis vector
        for semantic_axis_vector in self.semantic_axis_vector_list:
            # Calculate corpus bias for the semantic axis
            bias_for_semantic_axis_vector = self.bias_corpus_for_semantic_axis(semantic_axis_vector)
            # Calculate corpus intensity for the semantic axis
            intensity_for_semantic_axis_vector = self.intensity_corpus_for_semantic_axis(
                semantic_axis_vector, corpus_bias=bias_for_semantic_axis_vector)

            # Append bias and intensity results to their respective list
            self.bias_list.append(bias_for_semantic_axis_vector)
            self.intensity_list.append(intensity_for_semantic_axis_vector)

        # Create a pandas dataframe of the bias and intensity of every polar pair
        self.bias_intensity_df = pd.DataFrame({'polar_pairs': self.polar_pairs_list,
                                               'bias'       : self.bias_list, 
                                               'intensity'  : self.intensity_list})

        # Sort values of the dataframe by intensity decreasingly
        # Note that index of the DataFrame still mantain the same order than 
        # semantic_axis_vector_list for semantic axis
        self.bias_intensity_df = self.bias_intensity_df.sort_values(by='intensity', ascending=False)

        return self.bias_intensity_df


    def select_dim_by_intensity(self):
        """ This function selects a set number of dimensions with more intensity
        then they are transposed and inverted.
        If the number of the arbitrary dimensions is not set before, it is 10
        for default
        """

        # Check if bias and intensity have been calculated
        if self.bias_intensity_df == []:
            print("Calculate bias and intensity first calling calculate_bias_and_intensity()")  ## TO DO: Exception
            return

        # Since self.bias_intensity_df is sort decreasingly by intensity,
        # take the index of the first n_arbitrary_dimensions rows given that
        # index of the DataFrame still mantain semantic_axis_vector_list order
        self.dimensions_index = [self.bias_intensity_df.index[i]
                                 for i in range(self.n_arbitrary_dimensions)]

        # Create a direction matrix with the semantic axis vector corresponding 
        # to the indexes already collected
        self.dir_matrix = [self.semantic_axis_vector_list[index]
                      for index in self.dimensions_index]
        
        # Invert dir matrix
        self.__invert_dir_matrix()


    def generate_word_polar_embeddings(self):
        """ This functions calculates new polar embeddings for every word in corpus. 
        As a result the dictionary word_polar_embeddings has a numpy array containing
        the polar embedding for each word
        
        """

        # Check if direction matrix is transpose and inverted
        if self.dir_matrix_transpose_inverted == []:
            self.__calculate_dir_matrix_from_all_semantic_axis()

        # Go over every word in the word frequency dictionary
        for word in self.word_frequency_dictionary:

            # Multiply dir matrix transpose inverted to the word embedding of the
            # current word and add it to the dictionary of word polar dimensions 
            self.word_polar_embeddings[word] = np.matmul(
                self.dir_matrix_transpose_inverted, self.model[word])


    def generate_document_polar_embeddings(self):
        """ This functions calculates new polar embeddings for every document in corpus. 
        As a result the list document_polar_embeddings has a numpy array containing
        the polar embedding for each document in the same order that the documents are
        in the corpus
        """

        # Check if direction matrix is transpose and inverted
        if self.dir_matrix_transpose_inverted == []:
            self.__calculate_dir_matrix_from_all_semantic_axis()

        # Go over every document in corpus
        for document in self.corpus:

            # Initialize the vector representing the document
            doc_vec = np.zeros(300)
            doc_len = len(document)

            # For every word in the document, add its vector to the vector representing 
            # the document
            for word in document:
                doc_vec += np.array(self.model[word])

            # Divide the vector calculated in the length of the document, obtaining
            # the average word vector in the entire document
            doc_vec = doc_vec/doc_len
            
            # Multiply dir matrix transpose inverted to the vector of the
            # current document and add it to the dictionary of document polar dimensions 
            self.document_polar_embeddings.append(np.matmul(
                self.dir_matrix_transpose_inverted, doc_vec))


    def generate_corpus_polar_embedding(self):
        """ This functions calculates the polar embedding for the corpus.
        As a result corpus_polar_embedding is a numpy array with the value
        of the polar embedding that represents the corpus
        """

        # Check if direction matrix is transpose and inverted
        if self.dir_matrix_transpose_inverted == []:
            self.__calculate_dir_matrix_from_all_semantic_axis()

        # Initialize the vector representing the corpus
        corpus_vec = np.zeros(300)

        # For every word in the corpus, add its vector to the vector representing 
        # the corpus
        for document in self.corpus:
            for word in document:
                corpus_vec += np.array(self.model[word])
        
        # Divide the vector calculated in the length of the corpus, obtaining
        # the average word vector of the entire corpus
        corpus_vec = corpus_vec/self.corpus_len

        # Multiply dir matrix transpose inverted to the vector of the
        # corpus and save it
        self.corpus_polar_embedding = np.matmul(
            self.dir_matrix_transpose_inverted, corpus_vec)

    
    def generate_polar_embeddings(self):
        """ Generate polar embeddings for every word and document of the corpus and
        the corpus itself
        """        

        self.generate_word_polar_embeddings()
        self.generate_document_polar_embeddings()
        self.generate_corpus_polar_embedding()


    def cuantify_dim(self, on='corpus', embedding='word'):
        """ This function cuantify dimensions for documents or as a measure of complexity,
        where the words in the documents are represented with polar or embeddings. 
        It is first necessary to collect all embeddings for each document or words,
        then perform PCA with a certain percentage of represented variance
        wanted. The number of dimensions of the resultant vectors or principal components 
        are the values of complexity. Results are stored either in a list for documents
        case or in a int variable for corpus case.

        Args:
            on (str, optional): Describes if loading scores belongs to corpus or documents. 
                                Defaults to 'corpus'
            embedding (str, optional): Describes if the embeddings used are polar or word embeddings.
                                       Defaults to 'word'
        Returns:
            list(int) or int: In list case, list with all the number of dimensions needed to 
                represent a certain percentage of variance of each document in the corpus
                In int case, number of dimensions needed to represent a certain percentage of
                variance of the corpus
        """        

        # Check corpus is set
        if not self.corpus: 
            print('First set corpus')  ## TO DO: Exception
            return
        
        # Check cuantification is either for corpus or documents
        if on not in ['corpus', 'documents']:
            print('Dimensions can be cuantified either on corpus or documents')  ## TO DO: Exception
            return

        # Check embedding is either polar or word
        if embedding not in ['polar', 'word']:
            print('Embeddings can be either polar or word')  ## TO DO: Exception
            return

        # Before iterating over the corpus, each case  needs different actions
        if on == 'documents':
            if embedding == 'polar':
                # Check if words polar embeddings are already calculated
                if self.word_polar_embeddings == {}:
                    self.generate_word_polar_embeddings()
                
                # Clear previous results
                self.pca_documents_list.clear()
                self.n_dimensions_documents_list.clear()
            
            elif embedding == 'word':
                # Clear previous results
                self.pca_documents_list_WE.clear()
                self.n_dimensions_documents_list_WE.clear()

        elif on == 'corpus':
            # This is the only case that does not need to iterate over the corpus
            # Therefore, PCA is performed and it returns the number of dimensions
            if embedding == 'polar':
                # Check if document polar embeddings are already calculated
                if self.document_polar_embeddings == []:
                    self.generate_document_polar_embeddings()
                
                # Create PCA object
                self.pca_corpus = PCA(self.percentage_var)
                
                # Scale and center data before performing PCA
                scaled_data = preprocessing.StandardScaler().fit_transform(pd.DataFrame(self.get_document_polar_embeddings()))

                # Perform PCA and save the number of dimensions of the first resultant vector (it colud 
                # be any). Note that the resultant vectors are stored in the PCA object
                self.n_dimensions_corpus = len(self.pca_corpus.fit_transform(scaled_data)[0])

                return self.n_dimensions_corpus
            
            elif embedding == 'word':
                # List to stack word embedding representations of the documents
                documents_word_embeddings_lists = []

                # Create PCA object
                self.pca_corpus_WE = PCA(self.percentage_var)

    
        # Iterate over each document of the corpus
        for i, doc in enumerate(self.corpus):
            if on == 'documents':
                # Create a PCA object for each document and append it to a list of PCA objects
                # This list keeps the same order that the documents in the corpus for its PCA objects
                if embedding == 'polar': 
                    self.pca_documents_list.append(PCA(self.percentage_var))
                elif embedding == 'word': 
                    self.pca_documents_list_WE.append(PCA(self.percentage_var)) 

                # Collect all the embeddings for each word in the document
                # Note that the same embedding is appended as many times as its word appears
                # in the document
                embeddings_list = []
                for word in doc:
                    if embedding == 'polar':
                        embeddings_list.append(self.word_polar_embeddings[word])
                    elif embedding == 'word': 
                        embeddings_list.append(self.model[word])

                # Scale and center the data before performing PCA
                scaled_data = preprocessing.StandardScaler().fit_transform(pd.DataFrame(embeddings_list))

                # Perform PCA and append to n_dimensions_documents_list or 
                # n_dimensions_documents_list_WE the lenght of the first resultant vector
                # (it could be any vector). Note that the resultant vectors are stored in
                # the PCA objects saved in pca_documents_list or pca_documents_list_WE
                if embedding == 'polar':
                    n_dim = len(self.pca_documents_list[i].fit_transform(scaled_data)[0])
                    self.n_dimensions_documents_list.append(n_dim)
                elif embedding == 'word':
                    n_dim = len(self.pca_documents_list_WE[i].fit_transform(scaled_data)[0])
                    self.n_dimensions_documents_list_WE.append(n_dim)
            
            elif on == 'corpus' and embedding == 'word':
                # Initialize the vector representing the document
                doc_vec = np.zeros(300)
                doc_len = len(doc)

                # For every word in the document, add its vector to the vector representing 
                # the document
                for word in doc:
                    doc_vec += np.array(self.model[word])

                # Divide the vector calculated in the length of the document, obtaining
                # the average word vector in the entire document
                doc_vec = doc_vec/doc_len

                # Add it to the list of documents
                documents_word_embeddings_lists.append(doc_vec)

        # Return in documents cases
        if on == 'documents':
            return self.n_dimensions_documents_list if embedding == 'polar' else self.n_dimensions_documents_list_WE

        # Perform PCA and return in corpus case
        elif on == 'corpus' and embedding == 'word':

            # Scale and center data before performing PCA
            scaled_data = preprocessing.StandardScaler().fit_transform(pd.DataFrame(documents_word_embeddings_lists))

            # Perform PCA and save the number of dimensions of the first resultant vector (it colud 
            # be any). Note that the resultant vectors are stored in the PCA object
            self.n_dimensions_corpus_WE = len(self.pca_corpus_WE.fit_transform(scaled_data)[0])

            return self.n_dimensions_corpus_WE
                

    def calculate_loading_scores(self, on='corpus'):
        """ This functions put loading scores for every document PCA in a DataFrame and save
        it in the list loading_scores_documents. It also add a column for the sum of the 
        loading scores and a weighted sum column in wich every loading scores is multiplied
        for ratio of variance that represents the principal component

        Args:
            on (str, optional): Describes if loading scores belongs to corpus or documents. 
                                Defaults to 'corpus'
        """        

        # Check loading scores are for either corpus or documents
        if on not in ['corpus', 'documents']:
            print('Loading scores can be calculated either on corpus or documents')  ## TO DO: Exception
            return        
        
        if on == 'corpus':
            # Check if PCA has been performed
            if not self.pca_corpus:
                print('First call cuantify_dim to perform PCA on corpus using polar embeddings')  ## TO DO: Exception
                return

            # Taking PCA corpus objects
            pca_list = [self.pca_corpus]

        elif on == 'documents':
            # Clear previous results
            self.loading_scores_documents.clear()

            # Check if PCA has been performed
            if self.pca_documents_list == None:
                print('First call cuantify_dim to perform PCA on documents')  ## TO DO: Exception
                return

            # Taking PCA documents list
            pca_list = self.pca_documents_list
        
        # Iterate each pca object corresponding to every document
        for i, pca in enumerate(pca_list):

            # Create DataFrame with loading scores from PCA object
            loading_scores_df = pd.DataFrame(pca.components_.T, 
                                             index=self.polar_pairs_list, 
                                             columns=[f'PC_{ni+1}' for ni in range(len(pca.components_))])
            
            # Set a loading_scores_df copy as loading_scores_corpus or append it to loading_scores_documents
            if on == 'corpus':
                self.loading_scores_corpus = loading_scores_df.copy()
            elif on == 'documents':
                self.loading_scores_documents.append(loading_scores_df.copy())

            # Calculate weighted sum of the loading scores multiplying them for ratio of
            # variance that represents the principal component. 
            loading_scores_df = loading_scores_df * pca.explained_variance_ratio_
            loading_scores_df['weighted_sum'] = loading_scores_df.abs().sum(axis=1)

            # Add sum weighted sum to the saved DataFrame
            if on == 'corpus':
                self.loading_scores_corpus['weighted_sum'] = loading_scores_df['weighted_sum']
                self.loading_scores_corpus['sum'] = self.loading_scores_corpus.abs().sum(axis=1)
            elif on == 'documents':
                self.loading_scores_documents[i]['weighted_sum'] = loading_scores_df['weighted_sum']
                self.loading_scores_documents[i]['sum'] = self.loading_scores_documents[i].abs().sum(axis=1)

