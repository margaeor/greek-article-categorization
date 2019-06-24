from preprocessor import Preprocessor
import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
from functools import reduce


pool = ThreadPool(processes=3)



def cosine(x, y):
	res = np.arccos(np.dot(x, y) / (norm(x) * norm(y)))
	return res

def train_and_evaluate(prep,articles_train,articles_test,labels_train,labels_test):
	# Create dictionary
	print("Creating Dictionary...")
	dct = prep.create_word_dictionary(articles_train)
	print(len(dct))

	# Create tfidf table
	print("Creating tfidf matrix...")
	X_train = prep.create_tfidf_train(dct, articles_train, labels_train)
	X_test = prep.create_tfidf_test(dct, articles_test)

	# Tranform data
	print("Transforming data...")
	X_train = prep.transform_train(X_train, method='entropy')
	X_test = prep.transform_test(X_test)

	# Reduce dimensions
	print("Reducing dimensions...")
	# X_train = prep.reduce_dims_train(X_train,labels_train,'LDA',n_components=70,solver='svd')
	X_train = prep.reduce_dims_train(X_train, labels_train, 'PCA', n_components=120)
	X_test = prep.reduce_dims_test(X_test)

	# print(X_train.shape)

	# Encode labels
	print("Decoding Labels...")
	y_train = labels_train
	y_test = labels_test
	y_train = prep.encode_labels(labels_train)
	y_test = prep.encode_labels(labels_test)

	print("Training Model...")
	# prep.train_model(X_train,y_train,method='KNN',n_neighbors=5)
	# prep.train_model(X_train, y_train, method='KNN', n_neighbors=5,metric='mahalanobis',metric_params={'V': np.cov(X_train, rowvar=False)})
	prep.train_model(X_train, y_train, method='KNN', n_neighbors=5, metric=cosine)
	# prep.train_model(X_train, y_train, method='MEAN', metric=cosine)
	# prep.train_model(X_train, y_train, method='SVM', gamma='scale', decision_function_shape='ovo')
	# prep.train_model(X_train,y_train,method='NB')
	# prep.train_model(X_train, y_train, method='GMM', n_components=n, init_params='kmeans', covariance_type='full')
	#prep.train_model(X_train,y_train,method='GMM',n_components=1,init_params='kmeans',covariance_type='diag')
	# prep.train_model(X_train,y_train,method='RandomForest')
	#prep.train_model(X_train,y_train,method='ANN',epochs=80,batch_size=20,layers=[(50,'relu'),(20,'relu')])
	# prep.train_model(X_train,y_train,method='CNN')

	print("Evaluating model...")
	accuracy = prep.evaluate_model(X_test, y_test)
	print("Accuracy: ", accuracy)

	return accuracy

if __name__ == '__main__':

	prep = Preprocessor(ignore_pickles=True,strict=False,n_bigrams=3000,bigram_min_freq=5)

	# Preprocess
	print("Preprocessing...")
	articles, labels = prep.parse_files(('train','test'))

	N = 10000
	articles,labels = articles[:N],labels[:N]
	articles,labels = np.array(articles),np.array(labels)

	kf = KFold(n_splits=5,shuffle=True)

	accuracies = []

	i = 0
	for train_index,test_index in kf.split(articles,labels):

		articles_train, articles_test = articles[train_index], articles[test_index]
		labels_train, labels_test = labels[train_index], labels[test_index]

		print("----- [Pass "+str(i)+"] -------")

		async_acc = pool.apply_async(train_and_evaluate, (Preprocessor(ignore_pickles=True,strict=False),
														  articles_train,articles_test,labels_train,labels_test))


		accuracies.append(async_acc)

		i+=1

	#accuracies = np.array(accuracies)
	accuracies = np.array([acc.get() for acc in accuracies])
	#accuracies = np.array(map(lambda x: x.get(),accuracies))

	print("\n\n\nAverage accuracy ",100*np.mean(accuracies),"%")