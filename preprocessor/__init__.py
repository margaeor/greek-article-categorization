import re
import unicodedata
import os
import codecs
import pickle
import numpy as np
from keras import losses
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout
import sklearn

from numpy import inf

from keras.models import Sequential
from keras.layers import Dense

from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.ensemble import RandomForestClassifier as RFC

from sklearn import decomposition
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB
from sklearn.mixture import GaussianMixture as GMM

from greek_stemmer import GreekStemmer

np.warnings.filterwarnings('ignore')


class Preprocessor(object):


	def __init__(self):
		with open(os.path.join(
                  os.path.dirname(__file__), 'neutral_words.txt'), 'r',encoding="utf8") as fp:
			self.neutral_words = set([w[:-1] for w in fp.readlines()])
		#print(self.neutral_words)

		self.greek_stemmer = GreekStemmer()

		self.transform_model = None
		self.tranform_model_type = None

		self.reduction_model = None
		self.reduction_model_type = None

		self.classifier = None
		self.classifier_type = None

		self.label_dict = None

		self.idf = []

	def unpickle_data(self,file):
		#return []
		if os.path.isfile(file):
			with open(file, 'rb') as f:
				# The protocol version used is detected automatically, so we do not
				# have to specify it.
				return pickle.load(f)
		else:
			return []

	def pickle_data(self,file,data):
		#return
		with open(file, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


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

		pickle_file = './data/word_dict.pickle'

		if not recreate:
			data = self.unpickle_data(pickle_file)
			if len(data) > 0:
				words_dict,counter = data
				return words_dict

		for text in texts:
			for word in text:
				if word not in words_dict:
					words_dict[word] = counter
					counter += 1


		self.pickle_data(pickle_file,(words_dict,counter))

		return words_dict


	def create_tfidf_train(self,word_dict,texts):

		pickle_file = './data/train/tfidf.pickle'
		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			self.idf,tfidf = data
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

		idf = np.log(n_documents/np.sum(np.int32(m>0),axis=0))

		self.idf = idf

		tfidf = tft*idf

		self.pickle_data(pickle_file,(idf,tfidf))

		return tfidf

	def create_tfidf_test(self, word_dict, texts):

		if len(self.idf) == 0:
			raise Exception("You must create training idf first")

		pickle_file = './data/test/tfidf.pickle'
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

		pickle_file = './data/'+mode+'/tranform_'+method+'.pickle'

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

		self.pickle_data(pickle_file,l)

		return l

	def transform_test(self,tfidf):

		if self.transform_model == None:
			raise Exception("You must train first!")

		return self.transform_train(tfidf,self.transform_model,mode='test')


	def reduce_dims_train(self,X,y,method='PCA',**kwargs):

		pickle_file = './data/train/reduced_dims_'+method+'.pickle'

		data = self.unpickle_data(pickle_file)

		if len(data) > 0:
			l_kwargs,transform_model_type,reduction_model,transformed = data
			if transform_model_type == self.tranform_model_type and l_kwargs == kwargs:
				self.reduction_model = reduction_model
				return transformed

		if method == 'PCA':
			# n_components = 50
			self.reduction_model = decomposition.PCA(**kwargs)


		elif method == 'LDA':
			# n_components=50, solver=svd
			self.reduction_model = LDA(**kwargs)

		elif method == 'MEAN':
			# n_components=50
			self.reduction_model = MeanReduction(**kwargs)

		else:
			raise Exception("Wrong Method")


		self.reduction_model.fit(X, y)
		transformed = self.reduction_model.transform(X)

		self.pickle_data(pickle_file,(kwargs,self.tranform_model_type,self.reduction_model,transformed))

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
			self.classifier.fit(X, y)
		#77%
		elif method == 'NB':
			self.classifier = GaussianNB(**kwargs)
			y = np.argmax(y,axis=1)
			self.classifier.fit(X, y)

		elif method == 'GMM':
			# n_components=5, tol=1e-3, max_iter=100, init_params='kmeans'
			self.classifier = GMM(**kwargs)
			self.classifier.fit(X, y)
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
			self.classifier.add(Dense(y.shape[1], activation='softmax'))

			self.classifier.compile(loss=losses.kullback_leibler_divergence, optimizer='adam', metrics=['accuracy'])
			self.classifier.fit(X, y, epochs = n_epochs , batch_size = b_size )

		elif method == 'CNN':

			n_epochs = 50
			b_size = 10

			X = np.expand_dims(X,axis=2)

			model = Sequential()
			model.add(Convolution1D(nb_filter=512, filter_length=1, input_shape=(X.shape[1],1)))
			model.add(Activation('relu'))
			model.add(Flatten())
			model.add(Dropout(0.4))
			model.add(Dense(2048, activation='relu'))
			model.add(Dense(1024, activation='relu'))
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
		elif self.classifier_type == 'NB' or self.classifier_type == 'GMM':
			#print(pred.shape)
			pred = self.classifier.predict(X)
			y = np.argmax(y,axis=1)
			return np.sum(1*(pred == y))/len(y)
		else:
			pred = np.argmax(self.classifier.predict(X),axis=1)
			y = np.argmax(y,axis=1)
			print(pred.shape)
			return np.sum(1*(pred == y))/y.shape[0]

	#def pickle_data(self,file,object):
	#


class MeanReduction:

	def __init__(self,**kwargs):
		self.components = []
		self.n_components = kwargs['n_components'] if 'n_components' in kwargs else 50

	def fit(self,X,y):

		means = np.mean(X,axis=0)
		self.components = means.argsort()[::-1][:self.n_components]

	def transform(self,X):

		if len(self.components) == 0:
			raise Exception("You must train first!")

		return X[:,self.components]