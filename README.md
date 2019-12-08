# Data Mining Term Project (Fall 2019)
# Movie Search and Classifier

[Link to Project Proposal](https://ybardapurkar.github.io/Data_Mining_Project/project-proposal.html)

## Phase 1: Movie Search

[Link to Phase 1 report](https://ybardapurkar.github.io/Data_Mining_Project/phase-1.html)
[Link to Phase 1 demo](https://ybardapurkar.pythonanywhere.com/search)

## Phase 2: Classifier

[Link to Phase 2 report](https://ybardapurkar.github.io/Data_Mining_Project/phase-2.html)
[Link to Phase 2 demo](https://ybardapurkar.pythonanywhere.com/classify)

## Phase 3: Image Search and Captioning

[Link to Phase 3 report](https://ybardapurkar.github.io/Data_Mining_Project/phase-3.html)
[Link to Phase 3 demo (Image Search)](https://ybardapurkar.pythonanywhere.com/image_search)

## Setup on PythonAnywhere

1. Create an account on PythonAnywhere
2. Open up a bash console
3. Type in the console `git clone https://github.com/YBardapurkar/Data_Mining_Project.git`
4. Exit the terminal, and go to the "Web" tab to create a new web app
5. Create a "Manual" web app (not "Flask" webapp) with Python 3.7
6. Set "Source code" and "Working directory" to be `/home/yourname/Data_Mining_Project`
7. Open the WSGI configuration file, set the path variable as `path = '/home/yourname/Data_Mining_Project'` and uncomment the relevant parts under the heading "Flask". Change the last line to `from hello import app as application`, as currently the name of our main file is `hello.py`
8. Reload the web app, and it should be available at `https://yourname.pythonanywhere.com`.

## Setup on localhost

1. Open up a bash console
2. Type in `git clone https://github.com/YBardapurkar/Data_Mining_Project.git`
3. Enter the repository `cd Data_Mining_Project`
4. Install the required packages `pip install -r requirements.txt`
5. Run the app `python hello.py`
6. The application should be available on localhost at the default Flask port `127.0.0.1:5000`
