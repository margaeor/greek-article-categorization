import numpy as np
from numpy.linalg import norm

class MEAN_CLASSIFIER:
	def __init__(self, metric='cosine', metric_params=None,**kwargs):
		self.metric = metric
		self.dist = metric if not isinstance(metric,str) else self.distance()
		self.metric_params = metric_params
		self.X_train = np.array([])
		self.y_train = np.array([])
		self.mu = np.array([])

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
		self.n_classes = len(set(self.y_train))
		self.n_features = self.X_train.shape[1]

		mu = np.zeros((self.n_classes, self.n_features), dtype=np.float32)

		for i in range(self.n_classes):

			sample_mask = (self.y_train==i)

			# Calculate class mean
			mu[i,:] = np.mean(self.X_train[sample_mask,:],axis=0)

		self.mu = mu




	def predict(self,X):
		if len(self.X_train)<=0 or len(self.y_train)<=0 or len(self.y_train)!= len(self.X_train):
			raise Exception("Error with data")

		M = [[self.dist(m,x) for m in self.mu] for x in X]

		return np.argmin(M,axis=1)