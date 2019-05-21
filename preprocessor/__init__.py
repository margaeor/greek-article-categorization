import re
import unicodedata
import os
import codecs
import pickle
import numpy as np

from greek_stemmer import GreekStemmer

np.warnings.filterwarnings('ignore')


class Preprocessor(object):


	def __init__(self):
		with open(os.path.join(
                  os.path.dirname(__file__), 'neutral_words.txt'), 'r',encoding="utf8") as fp:
			self.neutral_words = set([w[:-1] for w in fp.readlines()])
		#print(self.neutral_words)

		self.greek_stemmer = GreekStemmer()




	def preprocess(self,text):

		text = self.strip_accents(text)

		text = text.upper()

		words = self.tokenize(text)

		words = [self.greek_stemmer.stem(w) for w in words if w not in self.neutral_words]

		words = [w for w in words if len(w)>0]

		return words


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



	def strip_accents(self, s):
		return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')



	def tokenize(self,text):

		text = re.sub(r'(([\w]+)\.)(([\w]+)\.)(([\w]+)\.)?(([\w]+)\.)?(([\w]+)\.)?','\2\4\6\8\10',text)
		text = re.sub(r'([^\w]|[0-9])+', ' ', text)

		words = text.split(' ')

		return words


	def create_tfidf(self,word_dict,texts):

		n_words = max(word_dict.values())
		n_documents = len(texts)

		m = []

		for text in texts:
			word_vec = [0]*(n_words+1)

			for word in text:

				word_vec[word_dict[word]] +=1

			m.append(word_vec)


		m = np.array(m)

		tft = m/np.sum(m,axis=1).reshape((-1,1))

		idf = np.log(n_documents/np.sum(np.int32(m>0),axis=0))

		#print(tft.shape)
		#print(idf.shape)

		tfidf = tft*idf

		#print(np.min(tfidf))

		return tfidf

	def transform(self,tfidf,method='entropy'):

		if method == 'entropy':

			p =  tfidf / np.sum(tfidf,axis=0)
			p[np.isnan(p)] = 1

			e = 1 + np.nan_to_num(np.sum(p*np.log(p),axis=0)/np.log(tfidf.shape[0]))

			l = e*np.log(1 + tfidf)

			return l



	def create_word_dictionary(self,texts,recreate=True):

		words_dict = {}
		counter = 0

		pickle_file = './data/word_dict.pickle'

		if not recreate and os.path.isfile(pickle_file):
			with open(pickle_file, 'rb') as f:
				# The protocol version used is detected automatically, so we do not
				# have to specify it.
				return pickle.load(f)

		for text in texts:
			for word in text:
				if word not in words_dict:
					words_dict[word] = counter
					counter += 1

		with open(pickle_file, 'wb') as f:
			# Pickle the 'data' dictionary using the highest protocol available.
			pickle.dump(words_dict, f, pickle.HIGHEST_PROTOCOL)

		return words_dict