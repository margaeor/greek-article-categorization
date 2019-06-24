import os
import re
import codecs
import shutil
import sys
import chardet

def convert_to_utf8(filename):
	with open(filename, 'rb') as opened_file:
		bytes_file = opened_file.read()
		chardet_data = chardet.detect(bytes_file)
		fileencoding = (chardet_data['encoding'])
		print('fileencoding', fileencoding)

		if fileencoding in ['utf-8', 'ascii']:
			print(filename + ' in UTF-8 encoding')
		else:
			# Convert file to UTF-8:
			# https://stackoverflow.com/q/19932116/5951529
			cyrillic_file = bytes_file.decode('cp1251')
			with codecs.open(filename, 'w', 'utf-8') as converted_file:
				converted_file.write(cyrillic_file)
			print(filename +
				  ' in ' +
				  fileencoding +
				  ' encoding automatically converted to UTF-8')


def parse_files(dir):
	articles = []
	labels = []
	for root, dirs, files in os.walk(dir):
		for name in files:

			link = os.path.join(root, name)

			if re.search(r'\.raw$',name):

				with codecs.open(link,'r',encoding='ISO-8859-7', errors='ignore') as f:

					m = re.match(r'^[a-zA-Z]+',name)
					if m:
						articles.append(f.read().replace('\n',' ').replace('\x96',' '))
						labels.append(m.group(0))


	if len(articles) != len(labels):
		raise Exception("Couldn't create labels")

	return articles,labels


#list_files('./data/train')