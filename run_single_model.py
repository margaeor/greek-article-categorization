from preprocessor import Preprocessor
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt




if __name__ == '__main__':

	available_models = ['NB','SVM','KNN','RandomForest','MEAN','GMM','ANN','CNN']
	available_transformations = ['entropy','binary','log','none']


	# Parse user input
	b1 = int(input('Choose an option:\n1)Use pickles\n2)Use only pickles for file parsing\n3)Ignore all pickles\n> '))
	assert b1 in range(1,4)

	b2 = int(input('Choose transformation method:\n'+"\n".join([str(i+1) + ")" + key for i, key in
															   enumerate(available_transformations)]) + "\n> "))-1
	assert b2 in range(0,len(available_transformations))

	b3 = int(input('Choose dimensionality reduction method:\n1)PCA\n2)LDA\n> '))

	assert b3 in range(1, 3)

	b4 = int(input('Choose model:\n' + "\n".join([str(i+1) + ")" + key
												  for i, key in enumerate(available_models)]) + "\n> "))-1
	assert b4 in range(0,len(available_models))

	prep = Preprocessor(ignore_pickles=(b1>1),strict=(b1==3),n_bigrams=3000,bigram_min_freq=5)

	# Preprocess
	print("Preprocessing...")
	articles_train, labels_train = prep.parse_files('train',is_train=True)
	articles_test, labels_test = prep.parse_files('test')

	# N = 10000
	# articles_train,labels_train = articles_train[:N],labels_train[:N]
	# articles_test,labels_test = articles_test[:N],labels_test[:N]


	# Create dictionary
	print("Creating Dictionary...")
	dct = prep.create_word_dictionary(articles_train)
	print("Dictionary length:",len(dct))


	# Create tfidf table
	print("Creating tfidf matrix...")
	X_train = prep.create_tfidf_train(dct,articles_train,labels_train)
	X_test = prep.create_tfidf_test(dct,articles_test)
	print("Number of features: ",X_train.shape[1])


	# Tranform data
	print("Transforming data...")
	X_train = prep.transform_train(X_train,method=available_transformations[b2])
	X_test = prep.transform_test(X_test)



	# Reduce dimensions
	print("Reducing dimensions...")
	if b3==2:
		X_train = prep.reduce_dims_train(X_train,labels_train,'LDA',n_components=6,solver='svd')
	else:
		X_train = prep.reduce_dims_train(X_train,labels_train,'PCA',n_components=120)

	X_test = prep.reduce_dims_test(X_test)
	print("Number of features after reduction: ", X_train.shape[1])


	# Encode labels
	print("Encoding Labels...")
	y_train = labels_train
	y_test = labels_test
	y_train = prep.encode_labels(labels_train)
	y_test = prep.encode_labels(labels_test)


	print("Training Model...")

	param_dict = {
		'NB': {'method': 'NB'},
		'SVM': {'method': 'SVM', 'kernel': 'rbf', 'C': 1.5, 'gamma': 'scale', 'decision_function_shape': 'ovo'},
		'RandomForest': {'method': 'RandomForest', 'n_estimators': 35},
		'MEAN': {'method': 'MEAN', 'metric': 'mahalanobis', 'metric_params': {'V': np.cov(X_train, rowvar=False)}},
		'GMM': {'method': 'GMM', 'covariance_type': 'diag', 'n_components': 11,'init_params':'kmeans'},
		'KNN': {'method': 'KNN', 'n_neighbors': 10, 'metric': 'cosine'},
		'ANN': {'method':'ANN','epochs': 40, 'batch_size': 20,'layers':[(50,'relu'),(20,'relu')]},
		'CNN': {'method':'CNN','epochs': 40, 'batch_size': 10},
	}

	prep.train_model(X_train,y_train,**param_dict[available_models[b4]])


	print("Evaluating model...")
	print("Accuracy: ",prep.evaluate_model(X_test,y_test))

