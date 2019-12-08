from datetime import datetime
from flask import Flask, render_template, request
from search import Search
from naivebayes import NaiveBayes
from imagesearch import ImageSearch
from imagecaption import ImageCaption
from sklearn.model_selection import train_test_split
from zipfile import ZipFile

with ZipFile('pre_processed_data.zip', 'r') as zipObj:
	zipObj.extractall()

search = Search();
search.init()

naive_bayes = NaiveBayes()
naive_bayes.init()

imagesearch = ImageSearch()
imagesearch.init()

image_caption = ImageCaption()
image_caption.init()

app = Flask(__name__)


@app.route("/")
def hello():
	return '<a href="/search">Search</a><br><a href="/classify">Classify</a><br><a href="/image_search">Image Search</a>'


@app.route("/search", methods = ['GET'])
def search_form():
	return render_template('search_form.html')


@app.route('/search', methods = ['POST'])
def search_result():
	query_string = request.form['query_string']

	# start
	start_time = datetime.now().timestamp()

	results, query_tokens = search.query_movie(query_string)

	# end
	total_time = (datetime.now().timestamp() - start_time)

	return render_template('search_form.html', query_string=query_string, query_tokens=query_tokens, res=results, runtime=total_time)

@app.route("/classify", methods = ['GET'])
def classify_form():
	return render_template('classify_form.html')


@app.route('/classify', methods = ['POST'])
def classify_result():
	query_string = request.form['query_string']

	# start
	start_time = datetime.now().timestamp()

	term_prob_table, results = naive_bayes.get_genres(query_string)

	# end
	total_time = (datetime.now().timestamp() - start_time)

	return render_template('classify_form.html', query_string=query_string, term_prob_table=term_prob_table, res=results, runtime=total_time)


@app.route("/image_search", methods = ['GET'])
def image_search_form():
	return render_template('image_search_form.html')


@app.route('/image_search', methods = ['POST'])
def image_search_result():
	query_string = request.form['query_string']

	# start
	start_time = datetime.now().timestamp()

	results, query_tokens = imagesearch.query_image(query_string)

	# end
	total_time = (datetime.now().timestamp() - start_time)

	return render_template('image_search_form.html', query_string=query_string, query_tokens=query_tokens, res=results, runtime=total_time)


@app.route("/image_caption", methods = ['GET'])
def image_caption_form():
	return render_template('image_caption_form.html')


@app.route('/image_caption', methods = ['POST'])
def image_image_caption_result():
	image_url = request.form['image_url']

	# start
	start_time = datetime.now().timestamp()

	caption = image_caption.generate_caption(image_url)
	results, query_tokens = imagesearch.query_image(caption)

	# end
	total_time = (datetime.now().timestamp() - start_time)

	print(image_url)
	print(caption)

	return render_template('image_caption_form.html', image_url=image_url, query_tokens=query_tokens, caption=caption, related_images=results, runtime=total_time)


if __name__ == "__main__":
	app.run()
