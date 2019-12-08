from collections import Counter, defaultdict
import json
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
import pandas as pd
from ProgressBar import ProgressBar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')

class NaiveBayesTest:

	def __init__(self):
		self.stop_words = stopwords.words('english')
		self.progress = ProgressBar()
		self.genres = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 
		'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 
		'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']
		
		# complete
		self.train, self.test = self.load_data('data/movies_metadata.csv')
		self.total_items = 0;
		self.class_frequency = defaultdict(int) # store classfrequency/ probability
		self.class_term_count = defaultdict(dict)  # to store each class has how may total term
		self.total_documents = 0
		self.unique_terms = set([]) # store number of unique term


	# return the dataframe
	def load_data(self, path):
		dataframe = pd.read_csv(path, low_memory=False)
		dataframe = dataframe[['id', 'genres', 'overview', 'title', 'original_title', 'poster_path']]
		dataframe.dropna(inplace=True)
		dataframe = dataframe[dataframe['genres'] != '[]']
		
		train, test = train_test_split(dataframe, test_size=0.1, random_state=2000)
		train.sort_index(inplace=True)
		test.sort_index(inplace=True)

		return train, test


	# return tokens of document 
	def tokenize(self, document):
		if pd.isnull(document):
			return []
		else:
			tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
			terms = tokenizer.tokenize(document.lower())
			filtered = [word for word in terms if not word in self.stop_words]
			return filtered


	# return stemmed tokens of document 
	def stem_tokenize(self, document):
		stemmer = PorterStemmer()
		tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

		terms = tokenizer.tokenize(document.lower())
		filtered = [stemmer.stem(word) for word in terms if not word in self.stop_words]
		return filtered


	# initialize
	def init(self):
		self.total_documents = len(self.train)
		# self.total_documents = 10000  # need to comment only for debuging

		# file_unique_tokens = open('classify_file_unique_tokens.pkl', 'rb')
		# file_postings = open('classify_file_postings.pkl', 'rb')
		# file_document_frequency = open('classify_file_document_frequency.pkl', 'rb')
		# file_lengths = open('classify_file_lengths.pkl', 'rb')

		for index, row in self.train.iterrows():
			if index == self.total_documents:
				break
			self.progress.progress_track(index, self.total_documents)

			row['genres'] = row['genres'].replace("'", '"')
			genres = json.loads(row['genres'])

			for item in genres:
				if item not in genres:
					continue

				current_class = item['name']
				if pd.isna(current_class):
					continue

				terms = self.stem_tokenize(row['overview'])

				self.class_frequency[current_class] = self.class_frequency[current_class] + 1
				self.total_items = self.total_items + 1

				for term in terms:
					if term not in self.class_term_count[current_class]:
						self.class_term_count[current_class][term] = 0
					self.class_term_count[current_class][term] = self.class_term_count[current_class][term] + 1

			# updating dictionary with all available terms
			self.unique_terms.update(set(terms))


	# return matching genres scores for the query
	def calculate_term_probability(self, query, parameter):
		terms = self.stem_tokenize(query)

		genre_prob = {}
		term_prob = {}
		query_prob = {}
		for genre in self.genres:
			genre_prob[genre] = self.class_frequency.get(genre, 0) / self.total_items

			for term in terms:
				if term not in term_prob:
					term_prob[term] = {}


				if term in self.class_term_count[genre]:
					term_prob[term][genre] = ((self.class_term_count[genre][term] + parameter * genre_prob[genre]) / (sum(list(self.class_term_count[genre].values()))) + parameter)
				else:
					term_prob[term][genre] = ((parameter * genre_prob[genre]) / (sum(list(self.class_term_count[genre].values()))) + parameter)


		for genre in self.genres:
			query_prob[genre] = genre_prob[genre]


		for term in term_prob:
			for genre in term_prob[term]:
				query_prob[genre] *= term_prob[term][genre]

		query_prob_counter = Counter(query_prob)
		query_prob_sum = 0
		for item in query_prob_counter:
			query_prob_sum += query_prob_counter[item]

		return query_prob_counter, terms


	def calculate_test_accuracy(self, parameter):
		score = 0
		total = 0
		i = 0
		self.progress = ProgressBar()
		print(self.test.shape[0])
		print('Calculate Accuracy: ')
		for index, row in self.test.iterrows():
			# if i == 100:
			# 	break
			row['genres'] = row['genres'].replace("'", '"')
			genres = json.loads(row['genres'])
			genres = [genre['name'] for genre in genres]
			# print('actual: ', str(set(genres)))

			try:
				res, terms = self.calculate_term_probability(row['overview'], parameter)
				# print(res)
			except:
				break
			# top_6 = list(dict(res.most_common(min(len(genres) * 3, 10))).keys())
			top_6 = list(dict(res.most_common(len(genres) + 2)).keys())
			# print('predicted: ', set(top_6))

			# for result in top_6:
			# 	if result in genres:
			# 		score += 1
			# 	total += 1
			for genre in genres:
				if genre in top_6:
					score += 1
				total += 1

			# score += len(set(genres).intersection(set(top_6))
			# total += len(set(genres))
			i += 1
			self.progress.progress_track(i, self.test.shape[0])
		print()
		print(score / total)
		return parameter, score / total
    
	def get_accuracy_report(self):
		# df = []
		df = pd.DataFrame(columns=['Accuracy'])
		i = 0.00000001
		while i <= 0.1:
			parameter, accuracy = self.calculate_test_accuracy(i)
			df.loc[parameter] = accuracy
			print(parameter, accuracy)
			i *= 10

		df.plot(kind='line')
		plt.xlabel('Smoothing parameter')
		plt.ylabel('Accuracy')
		plt.xscale('log')
		plt.title('Accuracy vs Smoothing parameter')
		plt.legend(loc='upper left')
		plt.show()
		print(df)

		return df


if __name__ == '__main__':
	naive_bayes_test = NaiveBayesTest()
	naive_bayes_test.init()
	df = naive_bayes_test.get_accuracy_report()

	df = pd.DataFrame(columns=['Accuracy'])

	df.plot(kind='line')
	plt.xlabel('Smoothing parameter')
	plt.ylabel('Accuracy')
	plt.xscale('log')
	plt.title('Accuracy vs Smoothing parameter')
	plt.legend(loc='upper left')
	plt.show()
	print(df)

'''

actual + 2:

              Accuracy
1.000000e-08  0.764758
1.000000e-07  0.781846
1.000000e-06  0.800155
1.000000e-05  0.809254
1.000000e-04  0.802597
1.000000e-03  0.794163
1.000000e-02  0.683977
1.000000e-01  0.598424


actual + 1:

              Accuracy
1.000000e-08  0.687084
1.000000e-07  0.706281
1.000000e-06  0.723924
1.000000e-05  0.725921
1.000000e-04  0.720040
1.000000e-03  0.712273
1.000000e-02  0.603972
1.000000e-01  0.533400
'''