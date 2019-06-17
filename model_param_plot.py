from plotting import Plotter
import pickle

if __name__ == '__main__':

	plotter = Plotter(threads=5)

	#plotter.visualize_bigrams(20)
	#print(plotter.prep.best_bigram_scores)

	#plotter.visualize_descriptive_terms(200)
	#print(plotter.prep.selected_words)

	methods = ['MEAN','GMM','RandomForest','SVM','KNN']

	for method in methods:
		print("Running " + method)
		getattr(plotter,method)(show=False)



	#plotter.run_all_kfold()

	#with open('a.out','rb') as f:
	#	a,b,c,d,e = pickle.load(f)
	#	plotter.plot_all_models(a,b,c,d,e)
	#	print("Hi")