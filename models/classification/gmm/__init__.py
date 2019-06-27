import numpy as np
from sklearn.mixture import GaussianMixture

class GMM:
	def __init__(self, n_components=5, covariance_type='full', tol=1e-3,
				 reg_covar=1e-6, max_iter=100,init_params='kmeans'):


		self.n_components = n_components
		self.covariance_type = covariance_type
		self.tol = tol
		self.reg_covar = reg_covar
		self.max_iter = max_iter
		self.init_params = init_params
		self.models = {}

	def fit(self,X,y):

		y = np.array(y)
		class_labels = set(y)
		self.n_classes = len(class_labels)
		self.P = np.zeros(self.n_classes, dtype=np.float32)

		# Fit one GMM to each class
		for i in range(self.n_classes):
			model = GaussianMixture(n_components=self.n_components,covariance_type=self.covariance_type,
									init_params=self.init_params,max_iter=self.max_iter)
			X_class = X[y==i,:]
			self.P[i] = np.sum(np.int32(X_class > 0)) / len(y)

			model.fit(X_class,y)
			self.models[i] = model


	def predict(self,X):

		probs = np.zeros((X.shape[0], self.n_classes))

		# Calculate the probability of a sample belonging to a certain class
		# by summing the log likelihoods of the individual gaussian components
		for (i,model) in self.models.items():
			probs[:,i] = np.sum(model._estimate_log_prob(X),axis=1)#+np.log(self.P[i])

		y_pred = np.argmax(probs,axis=1)

		return y_pred