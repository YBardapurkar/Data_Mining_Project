# Data Mining Term Project (Fall 2019)
# Movie Search and Classifier

[Link to Portfolio](https://yashbardapurkar.uta.cloud/portfolio/data-mining-project.html)

[Link to PythonAnywhere demo](https://ybardapurkar.pythonanywhere.com)

## Phase 1: Movie Search

### Setup on PythonAnywhere

1. Create an account on PythonAnywhere
2. Open up a bash console
3. Type in the console `git clone https://github.com/YBardapurkar/Data_Mining_Project.git`
4. Exit the terminal, and go to the "Web" tab to create a new web app
5. Create a "Manual" web app (not "Flask" webapp) with Python 3.7
6. Set "Source code" and "Working directory" to be `/home/yourname/Data_Mining_Project`
7. Open the WSGI configuration file, set the path variable as `path = '/home/yourname/Data_Mining_Project'` and uncomment the relevant parts under the heading "Flask". Change the last line to `from hello import app as application`, as currently the name of our main file is `hello.py`
8. Reload the web app, and it should be available at `https://yourname.pythonanywhere.com`.

### Setup on localhost

1. Open up a bash console
2. Type in `git clone https://github.com/YBardapurkar/Data_Mining_Project.git`
3. Enter the repository `cd Data_Mining_Project`
4. Install the required packages `pip install -r requirements.txt`
5. Run the app `python hello.py`
6. The application should be available on localhost at the default Flask port `127.0.0.1:5000`
