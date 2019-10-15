from datetime import datetime
from flask import Flask, render_template, request
from search import Search

search = Search();
search.init()
app = Flask(__name__)


@app.route("/")
def hello():
	return render_template('search_form.html')


@app.route('/search', methods = ['POST'])
def hello_name():
	query_string = request.form['query_string']

	# start
	start_time = datetime.now().timestamp()

	results, query_tokens = search.query_movie(query_string)

	# end
	total_time = (datetime.now().timestamp() - start_time)

	return render_template('search_form.html', query_string=query_string, query_tokens=query_tokens, res=results, runtime=total_time)

if __name__ == "__main__":
	app.run()
