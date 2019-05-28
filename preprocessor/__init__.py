import re
import unicodedata
import os
import codecs
import pickle
import numpy as np
import gensim

from keras import losses
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout

from models.classification.knn import KNN
from models.reduction.lda import LDA
#from models.reduction.pca import PCA
from models.classification.nb import NB
from models.classification.gmm import GMM

from sklearn.decomposition.pca import PCA
import sklearn
from sklearn.neighbors import DistanceMetric

from numpy import inf

from keras.models import Sequential
from keras.layers import Dense,MaxPooling1D

from sklearn.model_selection import GridSearchCV
from sklearn import svm
#from sklearn.neighbors import KNeighborsClassifier as KNN
#from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC

#from sklearn.decomposition import PCA
from sklearn import discriminant_analysis
from sklearn import decomposition
from sklearn.decomposition import LatentDirichletAllocation
#from sklearn.naive_bayes import GaussianNB
#from sklearn.mixture import GaussianMixture as GMM

from greek_stemmer import GreekStemmer

from nltk.collocations import *

np.warnings.filterwarnings('ignore')


class Preprocessor(object):


	def __init__(self,ignore_pickles=False):
		with open(os.path.join(
                  os.path.dirname(__file__), 'neutral_words.txt'), 'r',encoding="utf8") as fp:
			self.neutral_words = set([w[:-1] for w in fp.readlines()])
		#print(self.neutral_words)

		self.greek_stemmer = GreekStemmer()

		self.transform_model = None

		self.reduction_model = None
		self.reduction_model_type = None

		self.classifier = None
		self.classifier_type = None

		self.label_dict = None
		self.ignore_pickles = ignore_pickles
		self.idf = []

		self.id2word = None
		self.word_dict = None

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

	def pickle_data(self,file,data):

		data_file = file + '.data'
		meta_file = file + '.metadata'

		numpy_matrix = np.array([])
		metadata = []

		if self.ignore_pickles:
			return

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
				fh.write(numpy_matrix.data)




	def parse_files(self,dir,only_parse=False):

		dir = './data/'+dir

		pickle_file = dir+'/backup.pickle'

		if os.path.isfile(pickle_file):
			with open(pickle_file, 'rb') as f:
				# The protocol version used is detected automatically, so we do not
				# have to specify it.
				return pickle.load(f)

		articles = []
		labels = []
		for root, dirs, files in os.walk(dir):
			for name in files:

				link = os.path.join(root, name)

				if re.search(r'\.raw$', name):

					with codecs.open(link, 'r', encoding='ISO-8859-7', errors='ignore') as f:

						m = re.match(r'^[a-zA-Z]+', name)
						if m:
							data = f.read().replace('\n', ' ').replace('\x96', ' ')
							articles.append(data if only_parse else self.preprocess(data))
							labels.append(m.group(0))

		if len(articles) != len(labels):
			raise Exception("Couldn't create labels")

		with open(pickle_file, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump((articles,labels), f, pickle.HIGHEST_PROTOCOL)

		return articles, labels


	def preprocess(self,text):

		text = self.strip_accents(text)

		text = text.upper()

		words = self.tokenize(text)

		words = [self.greek_stemmer.stem(w) for w in words if w not in self.neutral_words]

		words = [w for w in words if len(w)>0]

		return words




	def strip_accents(self, s):
		return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')



	def tokenize(self,text):

		text = re.sub(r'(([\w]+)\.)(([\w]+)\.)(([\w]+)\.)?(([\w]+)\.)?(([\w]+)\.)?','\2\4\6\8\10',text)
		text = re.sub(r'([^\w]|[0-9])+', ' ', text)

		words = text.split(' ')

		return words


	def create_word_dictionary(self,texts,recreate=True):

		words_dict = {}
		counter = 0

		pickle_file = './data/word_dict'

		if not recreate:
			data = self.unpickle_data(pickle_file)
			if len(data) > 0:
				words_dict,counter = data
				self.id2word = {b:a for (a,b) in words_dict.items()}
				self.word_dict = words_dict
				return words_dict

		for text in texts:
			for word in text:
				if word not in words_dict:
					words_dict[word] = counter
					counter += 1

		self.pickle_data(pickle_file,(words_dict,counter))
		self.id2word = {b: a for (a, b) in words_dict.items()}
		self.word_dict = words_dict
		return words_dict


	def create_tfidf_train(self,word_dict,texts,doc_threshold=10,cutoff_percent=0.75):

		pickle_file = './data/train/tfidf'
		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			tfidf,self.idf = data
			return tfidf

		n_words = max(word_dict.values())
		n_documents = len(texts)

		m = []

		for text in texts:
			word_vec = [0]*(n_words+1)

			for word in text:

				if word in word_dict:
					word_vec[word_dict[word]] +=1

			m.append(word_vec)


		m = np.array(m)

		tft = m/np.sum(m,axis=1).reshape((-1,1))

		doc_frequency = np.sum(np.int32(m>0),axis=0)

		idf = np.log(n_documents/doc_frequency)

		idf = idf * np.int32(np.logical_and(doc_frequency > doc_threshold, doc_frequency<n_documents*cutoff_percent))

		self.idf = idf

		tfidf = tft*idf

		#self.calc_mutual_information(tfidf,m)

		self.pickle_data(pickle_file,(tfidf,idf))

		return tfidf

	def calc_mutual_information(self,tfidf,m):
		print("STARTING BAD THING")
		m1 = np.int32(m>0)

		red_model = MeanReduction(n_components=1000,word_dict=self.word_dict)
		red_model.fit(tfidf)

		m1 = red_model.transform(m1)

		res = [[0 if i > j else np.sum(np.int32(m1[:, i] == m1[:, j])) / m1.shape[0] for j in range(m1.shape[1])] for i in
		 range(m1.shape[1])]




		print(res)
		#print(res.shape)
		pass


	def create_tfidf_test(self, word_dict, texts):

		if len(self.idf) == 0:
			raise Exception("You must create training idf first")

		pickle_file = './data/test/tfidf'
		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			tfidf = data
			return tfidf


		n_words = max(word_dict.values())
		n_documents = len(texts)

		m = []

		for text in texts:
			word_vec = [0] * (n_words + 1)

			for word in text:

				if word in word_dict:
					word_vec[word_dict[word]] += 1

			m.append(word_vec)

		m = np.array(m)

		tft = m / np.sum(m, axis=1).reshape((-1, 1))

		tfidf = tft * self.idf

		self.pickle_data(pickle_file,tfidf)

		return tfidf

	def transform_train(self,tfidf,method='entropy',mode='train'):

		self.transform_model = method

		pickle_file = './data/'+mode+'/tranform_'+method

		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			return data

		l = []
		if method == 'entropy':

			p =  tfidf / np.sum(tfidf,axis=0)
			p[np.isnan(p)] = 1

			e = 1 + np.nan_to_num(np.sum(p*np.log(p),axis=0)/np.log(tfidf.shape[0]))
			print(e.shape)
			e[e== -inf] = 0

			l = e*np.log(1 + tfidf)


		elif method == 'binary':
			l = 1*(tfidf > 0)

		elif method == 'log':
			l = np.log(1+tfidf)

		#elif method == 'LDA':
		#	corpus_for_lda = [[(i,k) for (i,k) in enumerate(row)] for row in tfidf]
		#	lda_model_tfidf = gensim.models.LdaMulticore(corpus_for_lda, num_topics=6, id2word=self.id2word, passes=2,
		#												 workers=4)
		#	for idx, topic in lda_model_tfidf.print_topics(-1):
		#		print('Topic: {} Word: {}'.format(idx, topic))

		self.pickle_data(pickle_file,l)

		return l

	def transform_test(self,tfidf):

		if self.transform_model == None:
			raise Exception("You must train first!")

		return self.transform_train(tfidf,self.transform_model,mode='test')


	def reduce_dims_train(self,X,y,method='PCA',**kwargs):

		pickle_file = './data/train/reduced_dims_'+method

		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			transformed,l_kwargs,transform_model_type,reduction_model,= data
			if transform_model_type == self.transform_model and l_kwargs == kwargs:
				self.reduction_model = reduction_model
				print(transformed.shape)
				return transformed

		if method == 'PCA':
			# n_components = 50
			self.reduction_model = PCA(**kwargs)


		elif method == 'LDA':
			# n_components=50, solver=svd
			self.reduction_model = LDA(n_components=50)
			dct = {k: i for (i, k) in enumerate(set(y))}
			y = [dct[i] for i in y]

		elif method == 'MEAN':
			# n_components=50
			self.reduction_model = MeanReduction(**kwargs)

		else:
			raise Exception("Wrong Method")


		self.reduction_model.fit(X, y)
		transformed = self.reduction_model.transform(X)

		self.pickle_data(pickle_file,(transformed,kwargs,self.transform_model,self.reduction_model))

		return transformed


	def reduce_dims_test(self,X):

		if self.reduction_model == None:
			raise Exception("You must train first!")

		return self.reduction_model.transform(X)

	def encode_labels(self,y):

		s = set(y)

		n = len(s)

		if self.label_dict == None:
			self.label_dict = {l: [1 * (i == l) for l in range(n)] for (i, l) in enumerate(s)}

		y = np.array([self.label_dict[label] for label in y])

		return y


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
			#self.classifier.means_ = [np.mean(X[y == i,:],axis=0) for i in range(len(set(y)))]
			self.classifier.fit(X, y)
			#self.classifier._estimate_log_prob(X)
			#pass
		#79%
		elif method == 'RandomForest':
			self.classifier = RFC(**kwargs)
			self.classifier.fit(X, y)

		elif method == 'LDA':
			#n_components=5, learning_method='batch'|'online'
			self.classifier = LatentDirichletAllocation(**kwargs)
			self.classifier.fit(X, y)

		elif method == 'ANN':

			self.classifier = Sequential()
			l1,a1 = 12,'relu'
			l2,a2 = 8,'relu'
			n_epochs = 150
			b_size = 10

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

			self.classifier.compile(loss=losses.kullback_leibler_divergence, optimizer='adam', metrics=['accuracy'])
			self.classifier.fit(X, y, epochs = n_epochs , batch_size = b_size )

		elif method == 'CNN':

			n_epochs = 50
			b_size = 10

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
			model.compile(loss=losses.kullback_leibler_divergence, optimizer='adam', metrics=['accuracy'])
			model.fit(X, y, epochs = n_epochs , batch_size = b_size )

			self.classifier = model

		else:
			raise Exception("No such classifier exists!")



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

		#elif self.classifier_type == 'LDA':
		#	self.classifier.predict()
		elif self.classifier_type in ['NB','GMM','SVM','KNN']:
			#print(pred.shape)
			pred = self.classifier.predict(X)
			y = np.argmax(y,axis=1)
			return np.sum(1*(pred == y))/len(y)
		else:
			pred = np.argmax(self.classifier.predict(X),axis=1)
			y = np.argmax(y,axis=1)
			#print(pred.shape)
			return np.sum(1*(pred == y))/y.shape[0]

	#def pickle_data(self,file,object):
	#


class MeanReduction:

	def __init__(self,**kwargs):
		self.components = []
		self.word_dict = {} if 'word_dict' not in kwargs else kwargs['word_dict']
		self.n_components = kwargs['n_components'] if 'n_components' in kwargs else 50

	def fit(self,X,y=None):

		means = np.mean(X,axis=0)
		self.components = means.argsort()[::-1][:self.n_components]

		word_list = list(self.word_dict.items())
		self.reduced_dict = {word_list[i][0]:enum for (enum,i) in enumerate(self.components)}
		print(self.reduced_dict)

	def transform(self,X):

		if len(self.components) == 0:
			raise Exception("You must train first!")

		return X[:,self.components]