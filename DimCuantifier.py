import gensim
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.spatial.distance import cosine as scipy_cosine
from numpy import linalg
from collections import Counter
import matplotlib.pyplot as plt


class DimCuantifier:

    def __init__(self, model, pair_antonym_list):
        self.model = model
        self.pair_antonym_list = pair_antonym_list

        self.semantic_axis_vector_list = []
        self.corpus = []
        self.word_frequency_dictionary = Counter()
        self.bias_list = []
        self.intensity_list = []
        self.bias_intensity_df = None
        self.n_dimensions = 10
        self.dimensions_index = []
        self.dir_matrix_transpose_inverted = []
        self.word_dimensions = defaultdict()
        self.document_dimensions = []
        self.corpus_dimensions = None

        self.__generate_semantic_axis_vector_list()

    def __generate_semantic_axis_vector_list(self):
        for each_word_pair in self.pair_antonym_list:
            self.semantic_axis_vector_list.append(
                self.model[each_word_pair[0]] - self.model[each_word_pair[1]])
        self.semantic_axis_vector_list = np.array(
            self.semantic_axis_vector_list)

    def set_corpus(self, corpus):
        self.corpus = corpus
        self.__calculate_word_frequency_dictionary()

    def add_to_corpus(self, corpus):
        self.corpus = self.corpus + corpus
        self.__calculate_word_frequency_dictionary()

    def set_n_dimensions(self, k):
        self.n_dimensions = k
    
    def get_antonym_pair_list(self):
        return self.pair_antonym_list
    
    def get_semantic_axis_vector_list(self):
        return self.semantic_axis_vector_list

    def get_corpus(self):
        return self.corpus

    def get_word_frequency_dictionary(self):
        return self.word_frequency_dictionary

    def get_n_pair_antonym(self):
        return len(self.pair_antonym_list)
    
    def get_word_dimensions(self):
        return self.word_dimensions
    
    def get_document_dimensions(self):
        return self.document_dimensions
    
    def get_corpus_dimensions(self):
        return self.corpus_dimensions

    def contribution(self, word_vector, semantic_axis_vector):
        # check abs
        return 1.0 - scipy_cosine(word_vector, semantic_axis_vector)

    def __calculate_word_frequency_dictionary(self):

        if not self.corpus:
            print("Set corpus first")  # TO DO: Exception
            return

        self.word_frequency_dictionary.clear()

        for document in self.corpus:
            self.word_frequency_dictionary.update(document)

    def bias_corpus_for_semantic_axis(self, semantic_axis_vector):

        if not self.corpus:
            print("Set corpus first")  # TO DO: Exception
            return

        corpus_length = len(self.corpus)

        total_bias = 0.0
        for word in self.word_frequency_dictionary:
            total_bias += self.contribution(self.model[word],
                                            semantic_axis_vector) * self.word_frequency_dictionary[word]

        total_bias /= corpus_length

        return total_bias

    def intensity_corpus_for_semantic_axis(self, semantic_axis_vector, corpus_bias=None):

        if not self.corpus:
            print("Set corpus first")  # TO DO: Exception
            return

        if not corpus_bias:
            corpus_bias = self.bias_corpus_for_semantic_axis(
                semantic_axis_vector)

        corpus_length = len(self.corpus)

        total_intensity = 0.0
        for word in self.word_frequency_dictionary:
            total_intensity += ((self.contribution(
                self.model[word], semantic_axis_vector) - corpus_bias) ** 2) * self.word_frequency_dictionary[word]

        total_intensity /= corpus_length

        return total_intensity

    def calculate_bias_and_intensity(self):
        corpus_length = len(self.corpus)
        self.bias_list = []
        self.intensity_list = []

        for vector in self.semantic_axis_vector_list:
            B = self.bias_corpus_for_semantic_axis(vector)
            I = self.intensity_corpus_for_semantic_axis(vector, corpus_bias=B)

            self.bias_list.append(B)
            self.intensity_list.append(I)

        df = pd.DataFrame({'pair_words': self.pair_antonym_list,
                          'bias': self.bias_list, 'intensity': self.intensity_list})
        self.bias_intensity_df = df.sort_values(
            by='intensity', ascending=False)

        return self.bias_intensity_df

    def select_dim_by_intensity(self):
        if not self.n_dimensions:
            print('set number of dimensions first')  # TO DO: Exception
            return

        self.dimensions_index = [self.bias_intensity_df.index[i]
                                 for i in range(self.n_dimensions)]

        dir_matrix = [self.semantic_axis_vector_list[index]
                      for index in self.dimensions_index]

        self.dir_matrix_transpose_inverted = np.linalg.pinv(
            np.transpose(dir_matrix))

    def generate_word_dimensions(self, recalculate=True):
        if recalculate:
            self.word_dimensions.clear()

        for word in self.word_frequency_dictionary:
            self.word_dimensions[word] = np.matmul(
                self.dir_matrix_transpose_inverted, self.model[word])

    def generate_document_dimensions(self, recalculate=True):
        if recalculate:
            self.document_dimensions.clear()

        for document in self.corpus:
            doc_vec = np.zeros(300)
            doc_len = len(document)

            for word in document:
                doc_vec += np.array(self.model[word])

            doc_vec = doc_vec/doc_len
            
            self.document_dimensions.append(np.matmul(
                self.dir_matrix_transpose_inverted,doc_vec))

    def generate_corpus_dimensions(self):

        corpus_vec = np.zeros(300)
        corpus_len = 0

        for document in self.corpus:
            corpus_len += len(document)
            for word in document:
                corpus_vec += np.array(self.model[word])
        
        corpus_vec = corpus_vec/corpus_len

        self.corpus_dimensions = np.matmul(
            self.dir_matrix_transpose_inverted, corpus_vec)

    def generate_dimensions(self, recalculate=True):
        self.generate_word_dimensions(recalculate)
        self.generate_document_dimensions(recalculate)
        self.generate_corpus_dimensions()
    
    def plot_corpus_representation(self):
        fig, ax1 = plt.subplots(1,1, figsize=(5,5))
        plt.grid()
        
        color = 'blue'
        title = 'Corpus New Polar Dimensions Representation'
        top_val_dem = self.corpus_dimensions
        top_axis_dem = [self.pair_antonym_list[index] for index in self.dimensions_index]
        
        dem_max = max(top_val_dem)
        dem_min = min(top_val_dem)
        ax1.yaxis.tick_left()
        
        y = np.arange(len(top_val_dem))
        
        for i,_ in enumerate(top_val_dem):
            ax1.hlines(i, xmin=min(-dem_max,dem_min), xmax=max(-dem_min,dem_max), linewidth=2,color=color, zorder=1)
    
        ax1.scatter(top_val_dem, y, color=color, s=30, label='Polar Dimensions', zorder=2)
        ax1.set_yticks(y)
        ax1.set_yticklabels([item[0] for item in top_axis_dem], fontsize=15)
        ax1.set_ylim(-0.5,len(top_val_dem))
        ax1.vlines(0, ymin=-0.5, ymax=len(top_val_dem), linestyle='--', linewidth=1)

        ax_t1 = ax1.twinx()
        ax_t1.yaxis.tick_right()
        ax_t1.set_yticks(y)
        ax_t1.set_yticklabels([item[1] for item in top_axis_dem], fontsize=15)
        ax_t1.set_ylim(-0.5,len(top_val_dem))
        ax_t1.set_title(title, fontsize=15)

    def plot_document_representation(self):

        for i,_ in enumerate(self.corpus):
            fig, ax1 = plt.subplots(1,1, figsize=(5,5))
            plt.grid(axis = 'x')

            color = 'blue'
            title = f'Documents {i} New Polar Dimensions Representation'
            top_val_dem = self.document_dimensions[i]
            top_axis_dem = [self.pair_antonym_list[index] for index in self.dimensions_index]

            dem_max = max(top_val_dem)
            dem_min = min(top_val_dem)
            ax1.yaxis.tick_left()

            y = np.arange(len(top_val_dem))

            for i,_ in enumerate(top_val_dem):
                ax1.hlines(i, xmin=min(-dem_max,dem_min), xmax=max(-dem_min,dem_max), linewidth=2,color=color, zorder=1)

            ax1.scatter(top_val_dem, y, color=color, s=30, label='Polar Dimensions', zorder=2)
            ax1.set_yticks(y)
            ax1.set_yticklabels([item[0] for item in top_axis_dem], fontsize=15)
            ax1.set_ylim(-0.5,len(top_val_dem))
            ax1.vlines(0, ymin=-0.5, ymax=len(top_val_dem), linestyle='--', linewidth=1)

            ax_t1 = ax1.twinx()
            ax_t1.yaxis.tick_right()
            ax_t1.set_yticks(y)
            ax_t1.set_yticklabels([item[1] for item in top_axis_dem], fontsize=15)
            ax_t1.set_ylim(-0.5,len(top_val_dem))
            ax_t1.set_title(title, fontsize=15)

    def plot_word_representation(self):

        for i,word in enumerate(list(self.word_dimensions.keys())):
            fig, ax1 = plt.subplots(1,1, figsize=(5,5))
            plt.grid(axis = 'x')

            color = 'blue'
            title = f'Word \'{word}\' New Polar Dimensions Representation'
            top_val_dem = self.word_dimensions[word]
            top_axis_dem = [self.pair_antonym_list[index] for index in self.dimensions_index]

            dem_max = max(top_val_dem)
            dem_min = min(top_val_dem)
            ax1.yaxis.tick_left()

            y = np.arange(len(top_val_dem))

            for i,_ in enumerate(top_val_dem):
                ax1.hlines(i, xmin=min(-dem_max,dem_min), xmax=max(-dem_min,dem_max), linewidth=2,color=color, zorder=1)

            ax1.scatter(top_val_dem, y, color=color, s=30, label='Polar Dimensions', zorder=2)
            ax1.set_yticks(y)
            ax1.set_yticklabels([item[0] for item in top_axis_dem], fontsize=15)
            ax1.set_ylim(-0.5,len(top_val_dem))
            ax1.vlines(0, ymin=-0.5, ymax=len(top_val_dem), linestyle='--', linewidth=1)

            ax_t1 = ax1.twinx()
            ax_t1.yaxis.tick_right()
            ax_t1.set_yticks(y)
            ax_t1.set_yticklabels([item[1] for item in top_axis_dem], fontsize=15)
            ax_t1.set_ylim(-0.5,len(top_val_dem))
            ax_t1.set_title(title, fontsize=15)

