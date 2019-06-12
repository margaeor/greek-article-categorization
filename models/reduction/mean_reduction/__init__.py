import numpy as np

class MEAN_REDUCTION:

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