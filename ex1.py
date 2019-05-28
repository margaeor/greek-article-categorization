from preprocessor import Preprocessor
import numpy as np
from numpy.linalg import norm



def cosine(x, y):
	res = 1 - np.dot(x, y) / (norm(x) * norm(y))
	return res


if __name__ == '__main__':

	prep = Preprocessor(ignore_pickles=False)

	# Preprocess
	print("Preprocessing...")
	articles_train, labels_train = prep.parse_files('train')
	articles_test, labels_test = prep.parse_files('test')

	N = 10000
	articles_train,labels_train = articles_train[:N],labels_train[:N]
	articles_test,labels_test = articles_test[:N],labels_test[:N]

	# Create dictionary
	print("Creating Dictionary...")
	dct = prep.create_word_dictionary(articles_train)
	print(len(dct))

	# Create tfidf table
	print("Creating tfidf matrix...")
	X_train = prep.create_tfidf_train(dct,articles_train)
	X_test = prep.create_tfidf_test(dct,articles_test)

	# Tranform data
	print("Transforming data...")
	X_train = prep.transform_train(X_train,method='entropy')
	X_test = prep.transform_test(X_test)

	# Reduce dimensions
	print("Reducing dimensions...")
	#X_train = prep.reduce_dims_train(X_train,labels_train,'LDA',n_components=70,solver='svd')
	X_train = prep.reduce_dims_train(X_train,labels_train,'PCA',n_components=70)
	X_test = prep.reduce_dims_test(X_test)




	#print(X_train.shape)



	# Encode labels
	print("Decoding Labels...")
	y_train = labels_train
	y_test = labels_test
	y_train = prep.encode_labels(labels_train)
	y_test = prep.encode_labels(labels_test)




	print("Training Model...")
	#prep.train_model(X_train,y_train,method='KNN',n_neighbors=5)
	#prep.train_model(X_train, y_train, method='KNN', n_neighbors=5,metric='mahalanobis',metric_params={'V': np.cov(X_train, rowvar=False)})
	#prep.train_model(X_train, y_train, method='KNN', n_neighbors=5, metric=cosine)
	#prep.train_model(X_train,y_train,method='SVM', gamma='scale', decision_function_shape='ovo')
	#prep.train_model(X_train,y_train,method='NB')
	#prep.train_model(X_train, y_train, method='GMM', n_components=n, init_params='kmeans', covariance_type='full')
	#prep.train_model(X_train,y_train,method='GMM',n_components=30,init_params='kmeans',covariance_type='full')
	#prep.train_model(X_train,y_train,method='RandomForest')
	#prep.train_model(X_train,y_train,method='ANN',epochs=80,batch_size=20,layers=[(50,'relu'),(20,'relu')])
	#prep.train_model(X_train,y_train,method='CNN')

	for n in range(1,25):
		prep.train_model(X_train, y_train, method='GMM', n_components=n, init_params='kmeans', covariance_type='full')
		#print("Accuracy: ",prep.evaluate_model(X_test,y_test))
		print("Accuracy for "+str(n)+" is:", prep.evaluate_model(X_test, y_test))

	#print(X.shape)

	#print(prep.transform(tfidf))

	#print(tfidf.shape)
