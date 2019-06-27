import numpy as np

class PCA:

	def __init__(self,n_components):
		self.n_components = n_components


	def fit(self,X,y):


		# Use SVD to find principal components

		self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)

		self.components = self.V[:self.n_components]



	def transform(self,X):

		# Project data to our principal components

		return np.dot(X, self.components.T)