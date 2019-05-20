from preprocessor import Preprocessor



prep = Preprocessor()

articles, labels = prep.parse_files('train')

print(articles)