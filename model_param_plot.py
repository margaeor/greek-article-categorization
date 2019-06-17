from plotting import Plotter
import pickle
import time


if __name__ == '__main__':

	plotter = Plotter(threads=5,ignore_pickles=True)

	plotter.visualize_bigrams(40)
	print(plotter.prep.best_bigram_scores)

	#plotter.visualize_descriptive_terms(200)
	#print(plotter.prep.selected_words)

	# start = time.time()
	#
	# methods = ['MEAN','RandomForest','GMM','SVM','KNN']
	#
	# for method in methods:
	# 	print("Running " + method)
	# 	getattr(plotter,method)(show=True)
	#
	# end = time.time()
	# print(end - start," seconds")

	plotter.run_all_kfold()

	#with open('a.out','rb') as f:
	#	a,b,c,d,e = pickle.load(f)
	#	plotter.plot_all_models(a,b,c,d,e)
	#	print("Hi")