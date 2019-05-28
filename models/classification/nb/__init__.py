import numpy as np


class NB:
	def __init__(self):
		self.P = np.array([])
		self.mu = np.array([])
		self.vr = np.array([])

	def calculate_class_prob(self, X, mean, var):

		stdev = np.sqrt(var)

		exponent = np.exp(-(np.power(X - mean, 2) / (2 * np.power(stdev, 2))))

		gaussian = (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent

		return np.prod(gaussian,axis=1)


	def fit(self,X,y):

		y = np.array(y)
		class_labels = set(y)
		n_classes = len(class_labels)
		n_samples,n_features = X.shape

		self.P = np.zeros(n_classes, dtype=np.float32)
		self.mu = np.zeros((n_classes, n_features), dtype=np.float32)
		self.vr = np.zeros((n_classes, n_features), dtype=np.float32)

		for i in range(n_classes):
			label_mask = (y==i)
			X_class = X[label_mask,:]

			# Calculate a priori probability
			self.P[i] = np.sum(np.int32(X_class>0)) / len(y)

			# Calculate class mean
			self.mu[i, :] = np.mean(X_class, axis=0)

			# Calculate class variance
			self.vr[i, :] = np.var(X_class, axis=0)


	def predict(self,X):

		n_classes = self.P.shape[0]

		aposteriori = np.zeros((X.shape[0],n_classes))

		for i in range(n_classes):
			aposteriori[:,i] = 	self.calculate_class_prob(X,self.mu[i,:],self.vr[i,:]) * self.P[i]

		y_pred = np.argmax(aposteriori,axis=1)

		return y_pred
