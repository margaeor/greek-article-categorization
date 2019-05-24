from preprocessor import Preprocessor
import numpy as np



prep = Preprocessor()

# Preprocess
print("Preprocessing...")
articles_train, labels_train = prep.parse_files('train')
articles_test, labels_test = prep.parse_files('test')




#articles_train,labels_train = articles_train[:100],labels_train[:100]
#articles_test,labels_test = articles_test[:100],labels_test[:100]

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
#X_train = prep.reduce_dims_train(X_train,labels_train,'LDA',n_components=50,solver='svd')
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
prep.train_model(X_train,y_train,method='NB')
#prep.train_model(X_train,y_train,method='GMM',n_components=5,init_params='kmeans')
#prep.train_model(X_train,y_train,method='RandomForest')
#prep.train_model(X_train,y_train,method='ANN',layers=[(20,'relu'),(50,'relu')])
#prep.train_model(X_train,y_train,method='CNN')

print("Accuracy: ",prep.evaluate_model(X_test,y_test))

#print(X.shape)

#print(prep.transform(tfidf))

#print(tfidf.shape)
