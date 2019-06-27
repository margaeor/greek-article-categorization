from plotting import Plotter
import pickle
import time


if __name__ == '__main__':

	b1 = int(input('Enter number of threads(1-8, default=5):'))
	assert b1 in range(1,9)

	b2 = int(input('Choose action:\n1)Visualize Bigrams\n2)Visualize Terms\n3)Run model grid param search\n4)Run all '
				   'models kfold\n> '))
	assert b2 in range(1,5)

	b3 = int(input('Perform clean file parsing or use pickles?:\n1)Use pickles\n2)Perform clean file parsing\n> '))

	assert b3 in range(1,3)


	plotter = Plotter(threads=b1, ignore_pickles=True,strict=(b3==2))

	while b2 != 5:

		if b2 == 1:
			plotter.visualize_bigrams(40)
			#print(plotter.prep.best_bigram_scores)

		elif b2 == 2:
			plotter.visualize_descriptive_terms(200)
			#print(plotter.prep.selected_words)

		elif b2==3:

			start = time.time()

			methods = ['GMM','MEAN','RandomForest','SVM','KNN']

			for method in methods:
				print("Running " + method)
				getattr(plotter,method)(show=True)

			end = time.time()
			print(end - start," seconds")

		elif b2==4:

			start = time.time()

			plotter.run_all_kfold()

			end = time.time()

			print(end - start," seconds")

		b2 = int(
			input('\nChoose action:\n1)Visualize Bigrams\n2)Visualize Terms\n3)Run model grid param search\n4)Run all '
				  'models kfold\n5)Exit\n> '))
		assert b2 in range(1, 5)
