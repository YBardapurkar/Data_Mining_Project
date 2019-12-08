from collections import Counter, defaultdict
import json
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from operator import itemgetter
import pandas as pd
import pickle
from ProgressBar import ProgressBar
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')

class NaiveBayes:

	def __init__(self):
		self.stop_words = stopwords.words('english')
		self.progress = ProgressBar()
		self.genres = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 
		'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 
		'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']
		
		# complete
		self.data = self.load_data('data/movies_metadata.csv').fillna('')
		self.total_items = 0;
		self.class_frequency = defaultdict(int) # store classfrequency/ probability
		self.class_term_count = defaultdict(dict)  # to store each class has how may total term
		self.total_documents = 0
		self.unique_terms = set([]) # store number of unique term
		self.genre_prob = {}


	# return stemmed tokens of document 
	def stem_tokenize(self, document):
		# terms = document.lower().split()
		stemmer = PorterStemmer()
		tokenizer = RegexpTokenizer(r'[a-zA-Z]+')

		terms = tokenizer.tokenize(document.lower())
		filtered = [stemmer.stem(word) for word in terms if not word in self.stop_words]
		return filtered

	# return the dataframe
	def load_data(self, path):
		dataframe = pd.read_csv(path, low_memory=False)
		dataframe = dataframe[['id', 'genres', 'overview', 'title', 'original_title', 'poster_path']]
		dataframe.dropna(inplace=True)
		return dataframe[dataframe['genres'] != '[]']


	# return tokens of document 
	def tokenize(self, document):
		if pd.isnull(document):
			return []
		else:
			tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
			terms = tokenizer.tokenize(document.lower())
			filtered = [word for word in terms if not word in self.stop_words]
			return filtered

	# initialize
	def init(self):
		self.total_documents = len(self.data)
		# self.total_documents = 10000  # need to comment only for debuging

		# only when read pkl file, otherwise comment
		file_class_frequency = open('classify_file_class_frequency.pkl', 'rb')
		file_total_items = open('classify_file_total_items.pkl', 'rb')
		file_class_term_count = open('classify_file_class_term_count.pkl', 'rb')
		file_genre_prob = open('classify_file_genre_prob.pkl', 'rb')
		file_unique_terms = open('classify_file_unique_terms.pkl', 'rb')
		# ^

		for index, row in self.data.iterrows():
			if index == self.total_documents:
				break

			# only when read pkl file, otherwise comment
			data = pickle.load(file_class_frequency)
			self.class_frequency = data
			data = pickle.load(file_total_items)
			self.total_items = data
			data = pickle.load(file_class_term_count)
			self.class_term_count = data
			data = pickle.load(file_genre_prob)
			self.genre_prob = data
			data = pickle.load(file_unique_terms)
			self.unique_terms = data
			print('skip read')
			break
			# ^

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

		for genre in self.genres:
			self.genre_prob[genre] = self.class_frequency.get(genre, 0) / self.total_items

		# only when writing pkl file, otherwise comment
		# file_class_frequency = open('classify_file_class_frequency.pkl', 'wb')
		# file_total_items = open('classify_file_total_items.pkl', 'wb')
		# file_class_term_count = open('classify_file_class_term_count.pkl', 'wb')
		# file_genre_prob = open('classify_file_genre_prob.pkl', 'wb')
		# file_unique_terms = open('classify_file_unique_terms.pkl', 'wb')

		# pickle.dump(self.class_frequency, file_class_frequency)
		# pickle.dump(self.total_items, file_total_items)
		# pickle.dump(self.class_term_count, file_class_term_count)
		# pickle.dump(self.genre_prob, file_genre_prob)
		# pickle.dump(self.unique_terms, file_unique_terms)
		# ^

		file_class_frequency.close()
		file_total_items.close()
		file_class_term_count.close()
		file_genre_prob.close()
		file_unique_terms.close()
		

	# return matching genres scores for the query
	def calculate_term_probability(self, query):
		terms = self.stem_tokenize(query)
		parameter = 0.00001

		
		term_prob = {}
		query_prob = {}
		for genre in self.genres:
			# genre_prob[genre] = self.class_frequency.get(genre, 0) / self.total_items

			for term in terms:
				if term not in term_prob:
					term_prob[term] = {}


				if term in self.class_term_count[genre]:
					term_prob[term][genre] = ((self.class_term_count[genre][term] + parameter * self.genre_prob[genre]) / (sum(list(self.class_term_count[genre].values()))) + parameter)
				else:
					term_prob[term][genre] = ((parameter * self.genre_prob[genre]) / (sum(list(self.class_term_count[genre].values()))) + parameter)


		for genre in self.genres:
			query_prob[genre] = self.genre_prob[genre]


		for term in term_prob:
			for genre in term_prob[term]:
				query_prob[genre] *= term_prob[term][genre]

		query_prob_counter = Counter(query_prob)

		return term_prob, query_prob_counter

	# final result
	def get_genres(self, query):
		term_prob, query_prob_counter = self.calculate_term_probability(query)

		prob_sum = sum(list(query_prob_counter.values()))

		# print(query_prob_counter)
		res = []
		for item in query_prob_counter.most_common(6):
			# print(item[0] + ' : ' + str(item[1] / prob_sum))
			res.append([item[0], item[1] / prob_sum * 100])

		return self.get_term_prob_table(term_prob), res

	# return html table for term probabilities
	def get_term_prob_table(self, term_prob):
		table = ''
		table += '<table class="prob_table">'

		table += '<tr>'
		table += '<th></th>'

		for token in term_prob:
			table += '<th>' + token + '</th>'
		
		table += '</tr>'

		for genre in self.genres:
			table += '<tr>'
			table += '<td>' + genre + '</td>'
			for token in term_prob:
				table += '<td>' + '{:.8f}'.format(term_prob[token][genre]) + '</td>'
			table += '</tr>'

		table += '</table>'
		return table