import numpy as np
from numpy.linalg import norm
from collections import Counter
from multiprocessing.pool import ThreadPool

class KNN:
	def __init__(self, n_neighbors=5,metric='cosine', metric_params=None, n_jobs=None,**kwargs):
		self.metric = metric
		self.dist = metric if not isinstance(metric,str) else self.distance()
		self.metric_params = metric_params
		self.X_train = np.array([])
		self.y_train = np.array([])
		self.k = n_neighbors

		if isinstance(metric,str) and metric == 'mahalanobis':
			self.metric_params['VI'] = np.linalg.inv(self.metric_params['V'])

		pass

	def distance(self):

		def euclidean(U,v):

			return np.sqrt(np.sum(np.power(U - v, 2)))

		def cosine(x, y):
			res = 1 - np.dot(x, y) / (norm(x) * norm(y))
			return res

		def mahalanobis(x,y):
			VI = self.metric_params['VI']
			return np.sqrt(np.dot(np.dot((x - y), VI), (x - y).T))

		if isinstance(self.metric,str):
			return eval(self.metric)
		else:
			raise Exception("No such function!")


	def fit(self,X,y):
		self.X_train = X
		self.y_train = y

	def predict(self,X):
		if len(self.X_train)<=0 or len(self.y_train)<=0 or len(self.y_train)!= len(self.X_train):
			raise Exception("Error with data")

		# Calculate distances for every x in X between x and the training set
		dst = [[self.dist(X[i,:],self.X_train[j,:]) for j in range(self.X_train.shape[0])]
						  for i in range(X.shape[0])]


		k_neighbors = self.y_train[np.argsort(dst)[:,:self.k].flatten()].reshape((X.shape[0], -1))

		y_pred = [Counter(neighbors).most_common(1)[0][0] for neighbors in k_neighbors]

		return y_pred

