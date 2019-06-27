
from preprocessor import Preprocessor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score

from models.classification.knn import KNN
from models.classification.nb import NB
from models.classification.gmm import GMM
from models.classification.mean import MEAN_CLASSIFIER

from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import numpy as np
from multiprocessing.pool import ThreadPool
from wordcloud import WordCloud


class Plotter:

	def __init__(self,preprocess=True,threads=5,ignore_pickles=False,strict=False):

		# Initialize the thread pool used for parallel processing
		self.pool = ThreadPool(processes=threads)

		# Initialize the main preprocessor object
		self.prep = Preprocessor(ignore_pickles=ignore_pickles,strict=strict,n_bigrams=3000,bigram_min_freq=10)

		if preprocess:
			# Preprocess used for individual model plot
			self.preprocess()


	# Method used to call preprocessor so as to preprocess our data.
	# Used for individual model plot
	def preprocess(self,verbose=True):

		if verbose: print("Preprocessing...")
		thres = 4652


		self.articles_all, self.labels_all = self.prep.parse_files(('train','test'), is_train=True)
		self.articles_train, self.labels_train = self.articles_all[:thres],self.labels_all[:thres]
		self.articles_test, self.labels_test = self.articles_all[thres:],self.labels_all[thres:]


		# Create dictionary
		if verbose: print("Creating Dictionary...")
		self.dct = self.prep.create_word_dictionary(self.articles_train)

		# Create tfidf table
		if verbose: print("Creating tfidf matrix...")
		self.X_train = self.prep.create_tfidf_train(self.dct, self.articles_train, self.labels_train,n_dims=3000)
		self.X_test = self.prep.create_tfidf_test(self.dct, self.articles_test)


		# Tranform data
		if verbose: print("Transforming data...")
		self.X_train = self.prep.transform_train(self.X_train, method='binary')
		self.X_test = self.prep.transform_test(self.X_test)

		# Reduce dimensions
		if verbose: print("Reducing dimensions...")
		#self.X_train = self.prep.reduce_dims_train(self.X_train,self.labels_train,'LDA',n_components=70,solver='svd')
		self.X_train = self.prep.reduce_dims_train(self.X_train, self.labels_train, 'PCA', n_components=70)
		self.X_test = self.prep.reduce_dims_test(self.X_test)

		# Encoding the labels and convert them to numbers
		if verbose: print("Encoding Labels...")
		self.y_train = self.labels_train
		self.y_test = self.labels_test
		self.y_train = self.prep.encode_labels(self.labels_train)
		self.y_test = self.prep.encode_labels(self.labels_test)




	# Train and evaluate the accuracy of a model with certain parameters
	def __run_model(self, prep, X_train, y_train, X_test, y_test, params):

		print("Training "+params['method'])
		prep.train_model(X_train, y_train, **params)

		print("Evaluating "+params['method'])
		accuracy = prep.evaluate_model(X_test, y_test)

		return accuracy

	# Helper method used to train and evaluate all given models for a particular set
	# of training and test data
	def __train_and_evaluate_all(self,param_list, articles_train, articles_test, labels_train, labels_test,verbose=True):

		# Create dictionary
		if verbose: print("Creating Dictionary...")
		dct = self.prep.create_word_dictionary(articles_train)

		# Create tfidf table
		if verbose: print("Creating tfidf matrix...")
		X_train = self.prep.create_tfidf_train(dct, articles_train, labels_train)
		X_test = self.prep.create_tfidf_test(dct, articles_test)

		# Tranform data
		if verbose: print("Transforming data...")
		X_train = self.prep.transform_train(X_train, method='binary')
		X_test = self.prep.transform_test(X_test)

		# Reduce dimensions
		if verbose: print("Reducing dimensions...")
		#X_train = self.prep.reduce_dims_train(X_train,labels_train,'LDA',n_components=70,solver='svd')
		X_train = self.prep.reduce_dims_train(X_train, labels_train, 'PCA', n_components=70)
		X_test = self.prep.reduce_dims_test(X_test)


		if verbose: print("Decoding Labels...")
		y_train = self.prep.encode_labels(labels_train)
		y_test = self.prep.encode_labels(labels_test)


		if verbose: print("Training Models...")
		parallel = param_list[:-2]
		non_parallel = param_list[-2:]
		parallel_results,non_parallel_results = [],[]

		# Function handles returned by the parallel pool.
		# Used for models that support multi-threading (all models except ANN and CNN)
		parallel_handles = [self.pool.apply_async(self.__run_model, (Preprocessor(ignore_pickles=True, strict=False),
																	 X_train, y_train, X_test, y_test, param)) for param in parallel]

		# Run non-parallel models (ANN and CNN)
		for params in non_parallel:
			non_parallel_results.append(self.__run_model(self.prep, X_train, y_train, X_test, y_test, params))

		parallel_results = [handle.get() for handle in parallel_handles]

		return np.array(parallel_results+non_parallel_results)



	# Used to visualize the best bigrams using WordCloud
	def visualize_bigrams(self,limit=10,path='./report/img/bigrams.png'):

		word_dict=  self.prep.word_dict

		bigram_set = set({b[0]+" "+b[1] for b,s in self.prep.best_bigram_scores})
		bigram_dict = {bigram:self.prep.var[word_dict[bigram]] for bigram in bigram_set if bigram in self.prep.word_dict}

		bigram_list = list(bigram_dict.items())
		bigram_list = sorted(bigram_list,key=lambda x:-x[1])[:limit]
		bigram_dict = {a:b for a,b in bigram_list}

		wc = WordCloud(width = 2400, height = 2400,background_color ='white')
		wc_img = wc.generate_from_frequencies(bigram_dict)
		plt.figure(figsize=(10, 10))
		plt.imshow(wc_img, interpolation="bilinear")
		plt.axis("off")

		# Export image to file
		wc.to_file(path)
		plt.show()

	# Used to visualize the best terms using WordCloud
	def visualize_descriptive_terms(self,limit=10,path='./report/img/terms.png'):
		tmp = self.prep.selected_words
		tmp[0] = (tmp[0][0],tmp[0][1]/6)
		term_dict = {t:s for t,s in tmp[:limit]}
		wc = WordCloud(width = 2400, height = 2400,background_color ='white')
		wc_img = wc.generate_from_frequencies(term_dict)
		plt.figure(figsize=(10, 10))
		plt.imshow(wc_img, interpolation="bilinear")
		plt.axis("off")

		# Export image to file
		wc.to_file(path)
		plt.show()


	# Method used to run kfold validation for every classifier with specific params
	# and create a collective plot for all models to compare accuracy
	def run_all_kfold(self,path='./report/img/all_models.eps'):

		prep = Preprocessor(ignore_pickles=True, strict=False, n_bigrams=3000, bigram_min_freq=5)

		# Preprocess
		print("Preprocessing...")
		articles, labels = prep.parse_files(('train', 'test'))

		print("Number of articles:",len(articles))
		# N = 10000
		# articles, labels = articles[:N], labels[:N]
		articles, labels = np.array(articles), np.array(labels)

		# Initialize kfold validation
		kf = KFold(n_splits=5, shuffle=True)

		param_list = [
			{'method':'NB'},
			{'method':'SVM','kernel': 'rbf', 'C': 1.5, 'gamma': 'scale', 'decision_function_shape': 'ovo'},
			{'method':'RandomForest','n_estimators':35},
			{'method':'MEAN','metric': 'mahalanobis','metric_params':{'V': np.cov(self.X_train, rowvar=False)}},
			{'method':'GMM','covariance_type':'diag', 'n_components': 11},
			{'method':'KNN','n_neighbors': 10, 'metric': 'cosine'},
			{'method':'ANN','epochs': 40, 'batch_size': 20},
			{'method':'CNN','epochs': 40, 'batch_size': 10},
		]

		accuracies = []

		i = 0
		# Repeat for each kfold iteration
		for train_index, test_index in kf.split(articles, labels):

			# Get training and test data
			articles_train, articles_test = articles[train_index], articles[test_index]
			labels_train, labels_test = labels[train_index], labels[test_index]

			print("----- [Pass " + str(i+1) + "] -------")

			# Train and evaluate models
			results = self.__train_and_evaluate_all(param_list,articles_train,articles_test,labels_train,labels_test)

			accuracies.append(results)

			i += 1

		mean_accuracies = np.mean(np.array(accuracies),axis=0)

		# Display the accuracy of each model
		for param,accuracy in zip(param_list,list(mean_accuracies)):
			print("Model: "+param['method'],", Accuracy: %.2f%%" % (accuracy*100))

		labels = [params['method'] for params in param_list]

		excluded_params = ['method','gamma','decision_function_shape','metric_params']

		# Create a collective plot of all the models used for comparison
		self.__plot_all_models("All models",param_list,mean_accuracies,excluded_params,labels,path=path)


	# Try KNN model for different sets of parameters and create an accuracy plot
	def KNN(self,show=True):

		param_list = [
			{'n_neighbors': 5,'metric':'euclidean'},
			{'n_neighbors': 10, 'metric': 'euclidean'},
			{'n_neighbors': 20, 'metric': 'euclidean'},
			{'n_neighbors': 5, 'metric': 'cosine'},
			{'n_neighbors': 10, 'metric': 'cosine'},
			{'n_neighbors': 20, 'metric': 'cosine'},
			{'n_neighbors': 5, 'metric': 'mahalanobis','metric_params':{'V': np.cov(self.X_train, rowvar=False)}},
			{'n_neighbors': 10, 'metric': 'mahalanobis','metric_params':{'V': np.cov(self.X_train, rowvar=False)}},
			{'n_neighbors': 20, 'metric': 'mahalanobis','metric_params':{'V': np.cov(self.X_train, rowvar=False)}}
		]

		excluded_params = ['metric_params']

		name = 'KNN'

		accuracies = self.__grid_search(param_list, name)

		self.__plot_discrete_model(name, param_list, accuracies, excluded_params,show)


	# Try RandomForest model for different sets of parameters and create an accuracy plot
	def RandomForest(self,show=True):

		n_range = range(20, 80, 3)

		param_list = [
			[{'n_estimators': i} for i in n_range]
		]

		name = 'RandomForest'

		flattened_params = [a for params in param_list for a in params]

		accuracies = self.__grid_search(flattened_params, name)

		accuracies = np.array(accuracies)

		accuracies = accuracies.reshape(len(param_list), -1)

		excluded_params = ['metric_params']

		self.__plot_continuous_model(name, param_list, accuracies, excluded_params, 'n_estimators',show=show)


	# Try MEAN model for different sets of parameters and create an accuracy plot
	def MEAN(self,show=True):

		param_list = [
			{'metric':'euclidean'},
			{'metric': 'cosine'},
			{'metric': 'mahalanobis','metric_params':{'V': np.cov(self.X_train, rowvar=False)}},
		]

		name = 'MEAN'

		accuracies = self.__grid_search(param_list, name)
		excluded_params = ['metric_params']

		self.__plot_discrete_model(name, param_list, accuracies, excluded_params,show=show)

	# Try ANN model for different sets of parameters and create an accuracy plot
	def ANN(self,show=True):

		l_vals = np.arange(0.001,1,0.2)

		param_list = [
			[{'epochs': 10, 'batch_size': 10, 'learning_rate': i} for i in l_vals],
			[{'epochs': 10, 'batch_size': 20, 'learning_rate': i} for i in l_vals],
			[{'epochs': 10, 'batch_size': 50, 'learning_rate': i} for i in l_vals],
			[{'epochs': 10, 'batch_size': 80, 'learning_rate': i} for i in l_vals]
		]

		name = 'ANN'

		flattened_params = [a for params in param_list for a in params]

		accuracies = self.__grid_search(flattened_params, name, multi_threading=False)

		accuracies = np.array(accuracies)

		accuracies = accuracies.reshape(len(param_list),-1)

		excluded_params = ['metric_params','epochs']

		self.__plot_continuous_model(name,param_list,accuracies,excluded_params,'learning_rate',show=show)

	# Try CNN model for different sets of parameters and create an accuracy plot
	def CNN(self,show=True):

		l_vals = np.arange(0.001,1,0.2)

		param_list = [
			[{'epochs': 10, 'batch_size': 10, 'learning_rate': i} for i in l_vals],
			[{'epochs': 10, 'batch_size': 20, 'learning_rate': i} for i in l_vals],
			[{'epochs': 10, 'batch_size': 50, 'learning_rate': i} for i in l_vals],
			[{'epochs': 10, 'batch_size': 80, 'learning_rate': i} for i in l_vals]
		]

		name = 'CNN'

		flattened_params = [a for params in param_list for a in params]

		accuracies = self.__grid_search(flattened_params, name, multi_threading=False)

		accuracies = np.array(accuracies)

		accuracies = accuracies.reshape(len(param_list),-1)

		excluded_params = ['metric_params','epochs']

		self.__plot_continuous_model(name,param_list,accuracies,excluded_params,'learning_rate',show)

	# Try GMM model for different sets of parameters and create an accuracy plot
	def GMM(self,show=True):

		n_range = range(1,30)

		param_list = [
			[{'covariance_type':'full', 'n_components': i} for i in n_range],
			[{'covariance_type':'diag', 'n_components': i} for i in n_range],
		]

		name = 'GMM'

		flattened_params = [a for params in param_list for a in params]

		accuracies = self.__grid_search(flattened_params, name)

		accuracies = np.array(accuracies)

		accuracies = accuracies.reshape(len(param_list),-1)

		excluded_params = ['metric_params']

		self.__plot_continuous_model(name,param_list,accuracies,excluded_params,'n_components',show)

	# Try SVM model for different sets of parameters and create an accuracy plot
	def SVM(self,show=True):
		param_list = [
			{'kernel': 'rbf', 'C': 0.5,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'rbf', 'C': 1.0,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'rbf', 'C': 1.5,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'linear', 'C': 0.5,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'linear', 'C': 1.0,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'linear', 'C': 1.5,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'poly', 'C': 0.5,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'poly', 'C': 1.5,'gamma':'scale', 'decision_function_shape':'ovo'},
			{'kernel': 'poly', 'C': 1.0,'gamma':'scale', 'decision_function_shape':'ovo'},
		]

		name = 'SVM'

		accuracies = self.__grid_search(param_list, name)

		excluded_params = ['gamma','decision_function_shape']

		self.__plot_discrete_model(name, param_list, accuracies, excluded_params,show)


	# Try different combinations of parameters and measure accuracy for each
	# set of parameters.
	# Supports multi-threading for faster results.
	def __grid_search(self, param_list, name, multi_threading=True):

		accuracies = []

		for params in param_list:
			params['method'] = name
			prep = Preprocessor(ignore_pickles=True,n_bigrams=3000,bigram_min_freq=5)
			if multi_threading:
				accuracies.append(self.pool.apply_async(self.__train_and_evaluate, (prep, params)))
			else:
				accuracies.append(self.__train_and_evaluate(prep, params))

		if multi_threading:
			accuracies = [a.get() for a in accuracies]

		return accuracies

	# Private method used to make model accuracy plot for different model parameters
	def __plot_discrete_model(self, name, param_list, accuracies, excluded_params,show=True,path=None):

		excluded_params.append('method')

		path = './report/img/'+name + '.eps' if path is None else path

		y_labels = []
		for params in param_list:
			buff = ''
			for key in params:
				if key not in excluded_params:
					buff += key+'='+str(params[key])+'\n'
			y_labels.append(buff)

		#  create the figure
		fig, ax1 = plt.subplots(figsize=(9, 7))
		fig.subplots_adjust(left=0.215, right=0.88)
		fig.canvas.set_window_title(name)

		pos = np.arange(len(param_list))

		rects = ax1.barh(2*pos, [100*k for k in accuracies],
		 				 align='center',
						 height=1.5,tick_label=y_labels)

		ax1.set_title(name)

		ax1.set_xlim([0, 100])
		ax1.xaxis.grid(True, linestyle='--', which='major',
					   color='grey', alpha=.25)

		# Plot a solid vertical gridline to highlight the median position
		ax1.axvline(50, color='grey', alpha=0.25)

		# Set the right-hand Y-axis ticks and labels
		ax2 = ax1.twinx()

		# set the tick locations
		ax2.set_yticks(100*np.array(accuracies))
		# make sure that the limits are set equally on both yaxis so the
		# ticks line up
		ax2.set_ylim(ax1.get_ylim())

		# set the tick labels

		rect_labels = []
		for acc,rect in zip(accuracies,rects):
			# Rectangle widths are already integer-valued but are floating
			# type, so it helps to remove the trailing decimal point and 0 by
			# converting width to int type
			width = int(rect.get_width())

			rankStr = "%.1f %%" % (100*acc)

			xloc = -5
			# White on magenta
			clr = 'white'
			align = 'right'

			# Center the text vertically in the bar
			yloc = rect.get_y() + rect.get_height() / 2
			label = ax1.annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
								 textcoords="offset points",
								 ha=align, va='center',
								 color=clr, weight='bold', clip_on=True)
			rect_labels.append(label)

		ax1.set_xlabel('Accuracy')

		plt.savefig(path)

		if show: plt.show()


	# Private method used to create a line plot for the parameters of a model
	def __plot_continuous_model(self,name,param_list,accuracies,excluded_params,continuous_var,show=True,path=None):

		path = './report/img/'+name + '.eps' if path is None else path

		legend_texts = []

		for acc,params_set in zip(accuracies,param_list):

			n_vals = [param[continuous_var] for param in params_set]
			legend_text = "\n".join([key+"="+str(params_set[0][key]) for key in params_set[0] if key not in excluded_params
									 + [continuous_var,'method']])

			acc_n_vals = 100*np.array(acc)
			plt.plot(n_vals, acc_n_vals)
			legend_texts.append(legend_text)

		if len(legend_texts[0])>0:
			plt.legend(legend_texts)


		plt.ylabel('Prediction Accuracy')
		plt.xlabel(continuous_var)
		plt.title(name)

		plt.savefig(path)

		if show: plt.show()


	# Private method used to make a collective plot of all models and save it to files
	def __plot_all_models(self,name,param_list,accuracies,excluded_params,labels,show=True,path='./report/img/all_models.eps'):

		#print(param_list,accuracies)
		excluded_params.append('method')

		y_labels = []
		for params in param_list:
			buff = ''
			for key in params:
				if key not in excluded_params:
					buff += key+'='+str(params[key])+'\n'
			y_labels.append(buff)

		#  create the figure
		fig, ax1 = plt.subplots(figsize=(9, 7))
		fig.subplots_adjust(left=0.215, right=0.88)
		fig.canvas.set_window_title(name)

		pos = np.arange(len(param_list))

		rects = []
		for p,acc,y in zip(pos,accuracies,y_labels):
			rect = ax1.barh(2*p, 100*acc,
							 align='center',
							 height=1.5,tick_label="")

			rects.append(rect)

		ax1.set_title(name)

		ax1.set_xlim([0, 150])
		ax1.xaxis.grid(True, linestyle='--', which='major',
					   color='grey', alpha=.25)

		# Plot a solid vertical gridline to highlight the median position
		ax1.axvline(50, color='grey', alpha=0.25)

		# Set the right-hand Y-axis ticks and labels
		ax2 = ax1.twinx()

		# set the tick locations
		ax2.set_yticks(100*np.array(accuracies))
		# make sure that the limits are set equally on both yaxis so the
		# ticks line up
		ax2.set_ylim(ax1.get_ylim())


		rect_labels = []
		for acc,rect,y in zip(accuracies,rects,y_labels):
			# Rectangle widths are already integer-valued but are floating
			# type, so it helps to remove the trailing decimal point and 0 by
			# converting width to int type
			rect = rect[0]
			width = int(rect.get_width())

			rankStr = "%.1f %%" % (100*acc)

			xloc = -5
			# White on magenta
			clr = 'white'
			align = 'right'

			# Center the text vertically in the bar
			yloc = rect.get_y() + rect.get_height() / 2
			label = ax1.annotate(rankStr, xy=(width, yloc), xytext=(xloc, 0),
								 textcoords="offset points",
								 ha=align, va='center',
								 color=clr, weight='bold', clip_on=True)


			align = 'right'

			# Center the text vertically in the bar
			yloc = rect.get_y() + rect.get_height() / 2

			if y.count('\n') == 1:
				ax1.text(-40, yloc - 0.6, y)
			else:
				ax1.text(-40, yloc - 0.8, y)

			rect_labels.append(label)

		plt.legend(rects,labels)

		plt.savefig(path)

		ax1.set_xlabel('Accuracy')
		if show: plt.show()

	# Train and evaluate model with specific parameters
	def __train_and_evaluate(self, prep, params):

		prep.train_model(self.X_train, self.y_train, **params)

		return prep.evaluate_model(self.X_test,self.y_test)




