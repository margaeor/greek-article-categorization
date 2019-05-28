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
		n_samples,n_features = X.shape

		for i in range(self.n_classes):
			model = GaussianMixture(n_components=self.n_components,covariance_type=self.covariance_type,
									init_params=self.init_params,max_iter=self.max_iter)
			model.fit(X[y==i,:],y)
			self.models[i] = model


	def predict(self,X):

		probs = np.zeros((X.shape[0], self.n_classes))

		for (i,model) in self.models.items():
			probs[:,i] = np.sum(model._estimate_log_prob(X),axis=1)

		y_pred = np.argmax(probs,axis=1)

		return y_pred