from plotting import Plotter


if __name__ == '__main__':

	plotter = Plotter()

	methods = ['MEAN','RandomForest','GMM','SVM','KNN','ANN','CNN']

	for method in methods:
		getattr(plotter,method)()
		print("Running "+method)

	#plotter.CNN()