from preprocessor import Preprocessor
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt


def cosine(x, y):
	res = 1 - np.dot(x, y) / (norm(x) * norm(y))
	return res


if __name__ == '__main__':

	prep = Preprocessor(ignore_pickles=True,strict=True,n_bigrams=3000,bigram_min_freq=5)

	# Preprocess
	print("Preprocessing...")
	articles_train, labels_train = prep.parse_files('train',is_train=True)
	articles_test, labels_test = prep.parse_files('test')

	N = 10000
	articles_train,labels_train = articles_train[:N],labels_train[:N]
	articles_test,labels_test = articles_test[:N],labels_test[:N]

	#print(articles_train[2559])

	# Create dictionary
	print("Creating Dictionary...")
	dct = prep.create_word_dictionary(articles_train)
	print(len(dct))

	# Create tfidf table
	print("Creating tfidf matrix...")
	X_train = prep.create_tfidf_train(dct,articles_train,labels_train)
	X_test = prep.create_tfidf_test(dct,articles_test)

	# Tranform data
	print("Transforming data...")
	X_train = prep.transform_train(X_train,method='entropy')
	X_test = prep.transform_test(X_test)

	# Reduce dimensions
	print("Reducing dimensions...")
	#X_train = prep.reduce_dims_train(X_train,labels_train,'LDA',n_components=70,solver='svd')
	X_train = prep.reduce_dims_train(X_train,labels_train,'PCA',n_components=120)
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
	#prep.train_model(X_train, y_train, method='MEAN', metric=cosine)
	prep.train_model(X_train,y_train,method='SVM', gamma='scale', decision_function_shape='ovo')
	#prep.train_model(X_train,y_train,method='NB')
	#prep.train_model(X_train, y_train, method='GMM', n_components=n, init_params='kmeans', covariance_type='full')
	#prep.train_model(X_train,y_train,method='GMM',n_components=30,init_params='kmeans',covariance_type='full')
	#prep.train_model(X_train,y_train,method='RandomForest')
	#prep.train_model(X_train,y_train,method='ANN',epochs=80,batch_size=20,layers=[(50,'relu'),(20,'relu')])
	#prep.train_model(X_train,y_train,method='CNN')

	print("Evaluating model...")
	print("Accuracy: ",prep.evaluate_model(X_test,y_test))

	#exit(0)

	# acc_n_vals =[]
	# n_vals=[]
	#
	# for n in range(1,15):
	# 	prep.train_model(X_train, y_train, method='GMM', n_components=n, init_params='kmeans', covariance_type='diag')
	# 	#print("Accuracy: ",prep.evaluate_model(X_test,y_test))
	# 	acc = prep.evaluate_model(X_test, y_test)
	# 	n_vals.append(n)
	# 	acc_n_vals.append(acc)
	# 	print("Accuracy for "+str(n)+" is:",acc )
	#
	# acc_n_vals = np.array(acc_n_vals)
	#
	#
	# max_accuracy = np.max(acc_n_vals)
	# best_threshold = n_vals[np.argmax(acc_n_vals)]
	# print(max_accuracy, best_threshold)
	# plt.plot(n_vals, acc_n_vals)
	# plt.annotate('(%.2f,%.2f)' % (best_threshold, max_accuracy), xy=(best_threshold, max_accuracy),
	# 			 xytext=(best_threshold, max_accuracy + 0.3),
	# 			 arrowprops=dict(facecolor='black', shrink=0.05),
	# 			 ha='center'
	# 			 )
	# plt.ylabel('Prediction Accuracy')
	# plt.xlabel('n value (n)')
	# plt.ylim([0.3, 1.3])
	# plt.title('Accuracy for different values of n')
	# plt.show()

	#print(X.shape)

	#print(prep.transform(tfidf))

	#print(tfidf.shape)
