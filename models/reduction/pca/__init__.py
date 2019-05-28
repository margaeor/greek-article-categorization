import numpy as np

class PCA:

	def __init__(self,n_components):
		self.n_components = n_components


	def fit(self,X,y):

		#m,n = X.shape

		#self.mean = np.mean(X, axis=0)

		#X -= self.mean


		self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)

		self.components = self.V[:self.n_components]

		# R = np.cov(X,rowvar=False)

		#eigvals, eigvecs = np.linalg.eig(R)

		#eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

		# sort the eigvals in decreasing order
		#self.eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)


	def transform(self,X):

		#U, S, V = self.U,self.S,self.V
		#U = U[:, :self.n_components]

		# X_new = X * V = U * S * V^T * V = U * S
		#U *= S[:self.n_components]

		#return U

		return np.dot(X, self.components.T)

		#proj = np.array([item[1] for item in self.eiglist[:self.n_components]])

		#return np.matmul(X,proj.T)