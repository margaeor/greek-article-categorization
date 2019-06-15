from plotting import Plotter
import pickle

if __name__ == '__main__':

	plotter = Plotter()

	methods = ['MEAN','RandomForest','GMM','SVM','KNN','ANN','CNN']

	#for method in methods:
	#	getattr(plotter,method)()
	#	print("Running "+method)


	plotter.run_all_kfold()

	#with open('a.out','rb') as f:
	#	a,b,c,d,e = pickle.load(f)
	#	plotter.plot_all_models(a,b,c,d,e)
	#	print("Hi")