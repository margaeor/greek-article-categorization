from preprocessor import Preprocessor



prep = Preprocessor()

articles, labels = prep.parse_files('train')

articles = articles[:100]

dct = prep.create_word_dictionary(articles)

print(dct)

tfidf = prep.create_tfidf(dct,articles)

print(prep.transform(tfidf))

#print(tfidf.shape)



#print(articles)