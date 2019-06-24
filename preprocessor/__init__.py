import re
import unicodedata
import os
import codecs
import pickle
import nltk
import numpy as np
from numpy import inf

from itertools import chain
from keras.models import Sequential
from keras.layers import Dense,MaxPooling1D


from keras import losses
from keras import optimizers
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout


# Custom classification models

#from models.classification.knn import KNN
from models.classification.nb import NB
from models.classification.gmm import GMM
from models.classification.mean import MEAN_CLASSIFIER
#from models.reduction.mean_reduction import MEAN_REDUCTION
#from models.reduction.lda import LDA
#from models.reduction.pca import PCA


from sklearn.model_selection import GridSearchCV



from sklearn.neighbors import KNeighborsClassifier as KNN
#from sklearn.decomposition import PCA
#from sklearn.naive_bayes import GaussianNB
#from sklearn.mixture import GaussianMixture as GMM
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.decomposition.pca import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


from greek_stemmer import GreekStemmer
from nltk.collocations import *


np.warnings.filterwarnings('ignore')


class Preprocessor(object):


	def __init__(self,ignore_pickles=False,strict=False,n_bigrams=0,bigram_min_freq=3):

		self.greek_stemmer = GreekStemmer()

		with open(os.path.join(
				os.path.dirname(__file__), 'countries.txt'), 'r', encoding="utf8") as fp:
			self.countries = set([self.greek_stemmer.stem(self.strip_accents(w[:-1]).upper()) for w in fp.readlines()])

		with open(os.path.join(
                  os.path.dirname(__file__), 'neutral_words.txt'), 'r',encoding="utf8") as fp:
			self.neutral_words = set([w[:-1] for w in fp.readlines()])

		# Used to determine wether we should use parsing pickles or parse the files again
		self.strict = strict

		# Used to ignore preprocessing pickles
		self.ignore_pickles = ignore_pickles

		self.transform_model = None

		self.reduction_model = None
		self.reduction_model_type = None

		self.classifier = None
		self.classifier_type = None

		# Dictionary mapping labels to one-hot vectors
		self.label_dict = None

		# IDF calculated from training set
		self.idf = []

		# Dictionary mapping word ids to words
		self.id2word = None

		# Word dictionary mapping words to indexes
		self.word_dict = None

		# Indexes of most discriminating words
		self.selected_words = []

		# Variances of discriminating words
		self.var = np.array([])


		# Variables used for bigram features
		self.use_bigrams = n_bigrams>0
		self.n_bigrams = n_bigrams

		# Threshold for bigram occurence in texts
		self.bigram_min_freq = bigram_min_freq

		# List of bigram tuples
		self.bigrams = []

		# Scores of most descriptive bigrams
		self.best_bigram_scores = []
		self.bigram_indexes = []

		# Compile regexes that clear text
		self.clear_regexes =[

			# Merges decimal number into a single numerical value
			(re.compile(r'([0-9])(\s*[\.,]\s*)([0-9])'), '\1\3'),

			# Takes care of acronyms
			(re.compile(r'(([\w]+)\.)(([\w]+)\.)(([\w]+)\.)?(([\w]+)\.)?(([\w]+)\.)?'),'\2\4\6\8\10'),

			# Removes every character that is not a word or a digit
			(re.compile(r'([^\w0-9])+'), ' ')
		]

		# Constant used in place of country name
		self.COUNTRY_CONST = 'COUNTRYVALUE'

		# Constant used instead of numerical values
		self.NUMBER_CONST = 'NUMBERVALUE'


	# Unpickle data from file
	def unpickle_data(self,file):

		data_file = file + '.data'
		meta_file = file + '.metadata'

		if os.path.isfile(meta_file) and os.path.isfile(data_file) and not self.ignore_pickles:

			return_val = ()

			with open(meta_file, 'rb') as f:
				metadata = pickle.load(f)
				if isinstance(metadata,list) and len(metadata)>0:

					# If we have a numpy array
					if len(metadata[0]) == 3:
						dtype, w, h = metadata[0]

						with open(data_file, 'rb') as fh:
							return_val = (np.frombuffer(fh.read(), dtype=dtype).reshape((int(w), int(h))),)

					if len(metadata)>1:
						return_val = return_val + metadata[1]

			if len(return_val) == 1:
				return return_val[0]
			else:
				return return_val

		else:
			return []


	# Pickle data to file
	def pickle_data(self,file,data):

		data_file = file + '.data'
		meta_file = file + '.metadata'

		numpy_matrix = np.array([])
		metadata = []


		if type(data) == np.ndarray:
			numpy_matrix = data
			metadata.append((numpy_matrix.dtype, *numpy_matrix.shape))

		elif isinstance(data, tuple) and len(data)>0 and type(data[0]) == np.ndarray:
			numpy_matrix = data[0]
			metadata.append((numpy_matrix.dtype, *numpy_matrix.shape))
			if len(data) > 1:
				metadata.append(data[1:])
		else:
			metadata.append(())
			metadata.append(data)


		with open(meta_file, 'wb') as f:
			# Pickle the metadata file.
			pickle.dump(metadata, f, pickle.HIGHEST_PROTOCOL)

		# If we have metadata for numpy array, then write
		# the array to disc
		if len(metadata[0]) > 0:
			with open(data_file, 'wb+') as fh:
				fh.write(np.ascontiguousarray(numpy_matrix).data)



	def parse_files(self,dir,only_parse=False,is_train=False):

		is_kfold = isinstance(dir,tuple)
		is_train = is_kfold or is_train

		# List of directories to parse
		directories = list(dir) if is_kfold else [dir]

		# Pickle file for the given directories
		pickle_file = './data/'+"-".join(directories)+'.pickle'

		# If pickles exist and we don't want to ignore them, then load and return
		# preprocessed articles from pickle
		if not (self.ignore_pickles and self.strict) and os.path.isfile(pickle_file):
			with open(pickle_file, 'rb') as f:
				# The protocol version used is detected automatically, so we do not
				# have to specify it.
				self.best_bigram_scores,articles,labels = pickle.load(f)
				self.bigrams = set([b for b,s in self.best_bigram_scores])
				return articles,labels


		articles = []
		labels = []

		i=0

		# Loop through every directory
		for root, dirs, files in chain.from_iterable(os.walk('./data/'+dirr) for dirr in directories):

			# Loop through every file
			for name in files:

				link = os.path.join(root, name)

				# Open only .raw files
				if re.search(r'\.raw$', name):

					# Ignore encoding issues and open file as ISO-8859-7
					with codecs.open(link, 'r', encoding='ISO-8859-7', errors='ignore') as f:

						# Parse and preprocess file
						m = re.match(r'^[a-zA-Z]+', name)
						if m:
							data = f.read().replace('\n', ' ').replace('\x96', ' ')
							articles.append(data if only_parse else self.preprocess(data))
							labels.append(m.group(0))
							i+=1

		best_bigram_scores = []

		# If we want to use bigrams
		if self.use_bigrams:

			# Bigrams are collected from training set only
			if is_train:

				bigram_measures = nltk.collocations.BigramAssocMeasures()
				finder = BigramCollocationFinder.from_documents(articles)

				# Filter bigrams that appear in less than bigram_min_freq texts
				finder.apply_freq_filter(self.bigram_min_freq)

				# Get first n bigrams with highest PMI
				best_bigram_scores = [(b,s) for b, s in finder.score_ngrams(bigram_measures.pmi)[:self.n_bigrams]]
				best_bigrams = [b for b,s in best_bigram_scores]
				self.bigrams = set(best_bigrams)
				self.best_bigram_scores = best_bigram_scores

			# Append the bigrams to our article prepairing for feature extraction phase
			articles = [article + [b[0]+" "+b[1] for b in nltk.bigrams(article) if b in self.bigrams]
							for article in articles]

		if len(articles) != len(labels):
			raise Exception("Couldn't create labels")

		with open(pickle_file, 'wb') as f:
			# Pickle data to file
			pickle.dump((best_bigram_scores,articles,labels), f, pickle.HIGHEST_PROTOCOL)

		return articles, labels


	def preprocess(self,text):

		# Remove accent characters
		text = self.strip_accents(text)

		# Convert to uppercase
		text = text.upper()

		# Tokenize
		words = self.tokenize(text)

		# Stem words and remove neutral words
		words = [self.greek_stemmer.stem(w) for w in words if w not in self.neutral_words]

		r = re.compile('[0-9]')

		# Replace numerical values with NUMBER_CONST
		words = [self.NUMBER_CONST if bool(r.search(w)) else w for w in words]

		# Replace country names with COUNTRY_COST
		words = [self.COUNTRY_CONST if w in self.countries else w for w in words]

		# Remove words with less than 3 letters
		words = [w for w in words if len(w)>2]

		return words




	def strip_accents(self, s):
		return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')



	def tokenize(self,text):

		for regex,replacement in self.clear_regexes:
			text = regex.sub(replacement,text)

		words = text.split(' ')

		return words


	# Used to create a word dictionary
	def create_word_dictionary(self,texts,recreate=True):

		words_dict = {}
		counter = 0

		pickle_file = './data/word_dict'

		# Unpickle dict if possible
		data = self.unpickle_data(pickle_file)
		if len(data) > 0:
			words_dict,counter = data
			self.id2word = {b:a for (a,b) in words_dict.items()}
			self.word_dict = words_dict
			return words_dict

		# Loop through texts and add new words to dictionary
		for text in texts:
			for word in text:
				if word not in words_dict:
					words_dict[word] = counter
					counter += 1

		# Pickle dict
		self.pickle_data(pickle_file,(words_dict,counter))

		# Create inverse dict
		self.id2word = {b: a for (a, b) in words_dict.items()}
		self.word_dict = words_dict
		return words_dict


	def create_tfidf_train(self,word_dict,texts,labels,n_dims=3000):

		pickle_file = './data/train/tfidf'
		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			tfidf,self.var,self.selected_words,self.idf = data
			return tfidf

		# Useful values
		n_words = max(word_dict.values())
		n_documents = len(texts)
		label_set = set(labels)
		n_classes = len(label_set)
		n_features = n_words+1

		# Matrix used to represent number of occurences of each term in each text
		m = []

		# Loop through texts and create matrix m
		for text in texts:
			word_vec = [0]*(n_words+1)

			for word in text:

				if word in word_dict:
					word_vec[word_dict[word]] +=1

			m.append(word_vec)


		m = np.array(m)

		# Create y from labels
		dct = {k: i for (i, k) in enumerate(label_set)}
		y = np.array([dct[i] for i in labels])

		# For every class calculate the TF of every term
		m_class = np.zeros((n_classes,n_features))
		for i in range(n_classes):
			a = np.sum(m[y==i,:],axis=0)
			m_class[i,:] = a/np.sum(a)

		# Find the variance of tf among different classes
		# This is used to extract the terms that differentiate
		# most one class from the other
		var = np.var(m_class,axis=0)

		# Take top terms with largest variance among classes
		self.selected_dims = var.argsort()[-n_dims:][::-1]

		# Create tuples mapping term id to term variance
		self.selected_words = [(self.id2word[id],var[id]) for id in self.selected_dims]
		self.var = var

		# Reduced table m to new dimensions
		m_reduced = m[:,self.selected_dims]

		# Create tf matrix
		tft = m_reduced/np.sum(m_reduced,axis=1).reshape((-1,1))

		# Document frequency
		doc_frequency = np.sum(np.int32(m_reduced>0),axis=0)

		# Inverse document frequency
		idf = np.log(n_documents/doc_frequency)
		self.idf = idf

		# Calculate IDF
		tfidf = tft*idf

		# self.calc_mutual_information(tfidf,m_reduced)

		# Pickle our data
		self.pickle_data(pickle_file,(tfidf,var,self.selected_words,idf))

		return tfidf

	# Used to calculate mutual information
	def calc_mutual_information(self,tfidf,m,n_dims_reduced=1000):

		m1 = np.int32(m>0)

		n_documents = m1.shape[0]

		n_dims = m1.shape[1]

		res = [[np.NINF if i >= j else np.log(n_documents * np.sum(np.int32(m1[:, i] == m1[:, j]))/np.sum(np.int32(m[:,i]>0)) /
								 np.sum(np.int32(m[:,j]>0))) for j in range(m1.shape[1])] for i in range(m1.shape[1])]

		res = np.array(res) / np.log(2)

		res[np.isnan(res)] = np.NINF

		best_pairs = res.reshape(-1).argsort()[-n_dims_reduced:][::-1]

		results = [(a,b) for a,b in zip(best_pairs // n_dims, best_pairs % n_dims)]

		#for tup in results:
		#	print(self.id2word[self.selected_dims[tup[0]]]+"-"+self.id2word[self.selected_dims[tup[1]]])



	def create_tfidf_test(self, word_dict, texts):

		if len(self.idf) == 0:
			raise Exception("You must create training idf first")

		pickle_file = './data/test/tfidf'
		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			tfidf = data
			return tfidf


		n_words = max(word_dict.values())

		# Matrix used to represent number of occurences of each term in each text
		m = []

		# Loop through texts and extract features
		for text in texts:
			word_vec = np.array([0] * (n_words + 1))

			for word in text:

				if word in word_dict:
					word_vec[word_dict[word]] += 1

			m.append(word_vec)

		m = np.array(m)

		# Reduce matrix m according to the selected dimensions
		m_reduced = m[:,self.selected_dims]

		# Test set TF matrix
		tft = m_reduced / np.sum(m_reduced, axis=1).reshape((-1, 1))

		# Test set TFIDF matrix
		tfidf = tft * self.idf

		self.pickle_data(pickle_file,tfidf)

		return tfidf


	# Transform data
	def transform_train(self,tfidf,method='entropy',mode='train'):

		self.transform_model = method

		# If a corresponding pickle exists, load data from pickle
		pickle_file = './data/'+mode+'/tranform_'+method
		data = self.unpickle_data(pickle_file)
		if len(data) > 0:
			return data

		l = []

		# Entropy transformation
		if method == 'entropy':

			p =  tfidf / np.sum(tfidf,axis=0)
			p[np.isnan(p)] = 1

			e = 1 + np.nan_to_num(np.sum(p*np.log(p),axis=0)/np.log(tfidf.shape[0]))
			e[e== -inf] = 0

			l = e*np.log(1 + tfidf)

		# Binary transformation
		elif method == 'binary':
			l = 1*(tfidf > 0)

		# Logarithmic transformation
		elif method == 'log':
			l = np.log(1+tfidf)

		# No transformation
		elif method == 'none':
			l = tfidf

		self.pickle_data(pickle_file,l)

		return l

	# Transform test set using the chosen transformation method
	def transform_test(self,tfidf):

		if self.transform_model == None:
			raise Exception("You must train first!")

		return self.transform_train(tfidf,self.transform_model,mode='test')


	# Perform dimensionality reduction
	def reduce_dims_train(self,X,y,method='PCA',**kwargs):

		pickle_file = './data/train/reduced_dims_'+method

		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			transformed,l_kwargs,transform_model_type,reduction_model,= data
			if transform_model_type == self.transform_model and l_kwargs == kwargs:
				self.reduction_model = reduction_model
				return transformed

		if method == 'PCA':
			self.reduction_model = PCA(**kwargs)


		elif method == 'LDA':
			self.reduction_model = LDA(n_components=50)
			dct = {k: i for (i, k) in enumerate(set(y))}
			y = [dct[i] for i in y]

		else:
			raise Exception("Wrong Method")

		# Fit the reduction model to our data
		self.reduction_model.fit(X, y)

		# Perform dimensionality reduction
		transformed = self.reduction_model.transform(X)

		self.pickle_data(pickle_file,(transformed,kwargs,self.transform_model,self.reduction_model))

		return transformed

	# Perform dimensionality reduction to the test set using the chosen method
	def reduce_dims_test(self,X):

		if self.reduction_model == None:
			raise Exception("You must train first!")

		return self.reduction_model.transform(X)

	# Perform one-hot encoding to our labels
	def encode_labels(self,y):

		s = set(y)

		n = len(s)

		if self.label_dict == None:
			self.label_dict = {l: [1 * (i == l) for l in range(n)] for (i, l) in enumerate(s)}

		y = np.array([self.label_dict[label] for label in y])

		return y


	# Train model with different parameters and methods
	def train_model(self,X,y,method='KNN',**kwargs):

		self.classifier_type = method
		#85%
		if method == 'KNN':
			# n_neighbors=5, metric='minkowski'
			self.classifier = KNN(**kwargs)
			y = np.argmax(y, axis=1)
			self.classifier.fit(X, y)
		elif method == 'SVM':
			# gamma='scale', decision_function_shape='ovo'
			self.classifier = svm.SVC(**kwargs)
			y = np.argmax(y, axis=1)
			self.classifier.fit(X, y)
		#77%
		elif method == 'NB':
			self.classifier = NB(**kwargs)
			y = np.argmax(y,axis=1)
			self.classifier.fit(X, y)

		elif method == 'GMM':
			# n_components=5, tol=1e-3, max_iter=100, init_params='kmeans'
			self.classifier = GMM(**kwargs)
			y = np.argmax(y,axis=1)
			self.classifier.fit(X, y)
			#pass
		#79%
		elif method == 'RandomForest':
			self.classifier = RFC(**kwargs)
			self.classifier.fit(X, y)

		elif method == 'MEAN':
			self.classifier = MEAN_CLASSIFIER(**kwargs)
			y = np.argmax(y,axis=1)
			self.classifier.fit(X, y)

		elif method == 'ANN':

			self.classifier = Sequential()
			l1,a1 = 50,'relu'
			l2,a2 = 20,'relu'
			learning_rate = 0.001
			n_epochs = 50
			b_size = 10

			if 'learning_rate' in kwargs:
				learning_rate = kwargs['learning_rate']

			if 'layers' in kwargs:
				l1,a1 = kwargs['layers'][0]
				l2,a2 = kwargs['layers'][1]

			if 'epochs' in kwargs:
				n_epochs = kwargs['epochs']

			if 'batch_size' in kwargs:
				b_size = kwargs['batch_size']

			self.classifier.add(Dense(l1, input_dim=X.shape[1], activation=a1))
			self.classifier.add(Dense(l2, activation=a2))
			self.classifier.add(Dense(30, activation=a2))
			self.classifier.add(Dense(y.shape[1], activation='softmax'))

			optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
			self.classifier.compile(loss=losses.kullback_leibler_divergence, optimizer=optimizer, metrics=['accuracy'])
			self.classifier.fit(X, y, epochs = n_epochs , batch_size = b_size )

		elif method == 'CNN':

			n_epochs = 50
			b_size = 10
			learning_rate = 0.001


			if 'learning_rate' in kwargs:
				learning_rate = kwargs['learning_rate']

			if 'epochs' in kwargs:
				n_epochs = kwargs['epochs']

			if 'batch_size' in kwargs:
				b_size = kwargs['batch_size']

			X = np.expand_dims(X,axis=2)

			model = Sequential()
			model.add(Convolution1D(nb_filter=128, filter_length=1, input_shape=(X.shape[1],1)))
			model.add(MaxPooling1D(pool_size=2, strides=None, padding='valid'))
			model.add(Activation('relu'))
			model.add(Flatten())
			model.add(Dropout(0.4))
			model.add(Dense(128, activation='relu'))
			model.add(Dense(64, activation='relu'))
			model.add(Dense(y.shape[1]))
			model.add(Activation('softmax'))

			optimizer = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999)
			model.compile(loss=losses.kullback_leibler_divergence, optimizer=optimizer, metrics=['accuracy'])
			model.fit(X, y, epochs = n_epochs , batch_size = b_size )

			self.classifier = model

		else:
			raise Exception("No such classifier exists!")


	# Evaluate our model
	def evaluate_model(self,X,y):

		if self.classifier==None or self.classifier_type == None:
			raise Exception("You must first train the classifier!")

		if self.classifier_type == 'ANN':
			loss,acc = self.classifier.evaluate(X, y)

			return acc
		elif self.classifier_type == 'CNN':
			X = np.expand_dims(X, axis=2)

			loss, acc = self.classifier.evaluate(X, y)

			return acc

		elif self.classifier_type in ['NB','GMM','SVM','KNN','MEAN']:
			pred = self.classifier.predict(X)
			y = np.argmax(y,axis=1)
			return np.sum(1*(pred == y))/len(y)
		else:
			pred = np.argmax(self.classifier.predict(X),axis=1)
			y = np.argmax(y,axis=1)
			return np.sum(1*(pred == y))/y.shape[0]

