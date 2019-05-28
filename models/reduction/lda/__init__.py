import numpy as np

class LDA:

	def __init__(self, solver='svd', shrinkage=None, priors=None,
				 n_components=None, store_covariance=False, tol=1e-4):
		self.solver = solver
		self.shrinkage = shrinkage
		self.priors = priors
		self.n_components = n_components
		self.store_covariance = store_covariance  # used only in svd solver
		self.tol = tol  # used only in svd solver
		self.eiglist = np.array([])

	def fit(self,X,y):

		num_of_classes = len(set(y))

		#self.n_components = min(self.n_components,num_of_classes-1)

		y = np.array(y)

		num_features = X.shape[1]

		P = np.zeros(num_of_classes,dtype=np.float32)
		mu = np.zeros((num_of_classes, num_features),dtype=np.float32)
		Sw = np.zeros((num_features,num_features),dtype=np.float32)
		Sb = np.zeros((num_features,num_features),dtype=np.float32)

		m0 = np.mean(X, axis=0)

		for i in range(num_of_classes):

			sample_mask = (y==i)

			# Calculate a priori probability
			P[i] = np.sum(np.int32(sample_mask))/len(y)

			# Calculate class mean
			mu[i,:] = np.mean(X[sample_mask,:],axis=0)

			# Calculate within class scatter matrix
			Sw +=  P[i] * np.cov(X[sample_mask,:].T)

			# Calculate between class scatter matrix
			Sb +=  P[i] * np.matmul((mu[i,:]-m0).T,mu[i,:]-m0)

		mat= np.dot(np.linalg.pinv(Sw),Sb)

		eigvals, eigvecs = np.linalg.eig(mat)

		eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

		# sort the eigvals in decreasing order
		self.eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

	def transform(self,X):

		proj = np.array([item[1] for item in self.eiglist[:self.n_components]])

		res = np.matmul(X,proj.T)

		#print(res.shape)

		return res


	def estimate_params(self,data):
		# group data by label column
		grouped = data.groupby(self.data.ix[:, self.labelcol])

		# calculate means for each class
		means = {}
		for c in self.classes:
			means[c] = np.array(self.drop_col(self.classwise[c], self.labelcol).mean(axis=0))

		# calculate the overall mean of all the data
		overall_mean = np.array(self.drop_col(data, self.labelcol).mean(axis=0))

		# calculate between class covariance matrix
		# S_B = \sigma{N_i (m_i - m) (m_i - m).T}
		S_B = np.zeros((data.shape[1] - 1, data.shape[1] - 1))
		for c in means.keys():
			S_B += np.multiply(len(self.classwise[c]),
							   np.outer((means[c] - overall_mean),
										(means[c] - overall_mean)))

		# calculate within class covariance matrix
		# S_W = \sigma{S_i}
		# S_i = \sigma{(x - m_i) (x - m_i).T}
		S_W = np.zeros(S_B.shape)
		for c in self.classes:
			tmp = np.subtract(self.drop_col(self.classwise[c], self.labelcol).T, np.expand_dims(means[c], axis=1))
			S_W = np.add(np.dot(tmp, tmp.T), S_W)

		# objective : find eigenvalue, eigenvector pairs for inv(S_W).S_B
		mat = np.dot(np.linalg.pinv(S_W), S_B)
		eigvals, eigvecs = np.linalg.eig(mat)
		eiglist = [(eigvals[i], eigvecs[:, i]) for i in range(len(eigvals))]

		# sort the eigvals in decreasing order
		eiglist = sorted(eiglist, key=lambda x: x[0], reverse=True)

		# take the first num_dims eigvectors
		w = np.array([eiglist[i][1] for i in range(self.num_dims)])

		self.w = w
		self.means = means
		return