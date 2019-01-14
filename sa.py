from flask import Flask, render_template, request, redirect, url_for, session
from flask_session import Session
from form import MyForm
from logging import FileHandler, WARNING

import pickle
import pandas as pd

app = Flask(__name__)

file_handler = FileHandler('errorlog.txt') #file_handler logs errors
file_handler.setLevel(WARNING)
app.config.from_object(__name__) # Config for Flask-Session

app.config['MAX_CONTENT_LENGTH'] = 1 * 1024 * 1024 #max upload size is 1MB
app.config['SECRET_KEY'] = 'secret'
app.config['ALLOWED_EXTENSIONS'] = set(['txt'])#restricts file extensions to the .txt extension
app.config['SESSION_TYPE'] = 'filesystem' #config for Flask Session, indicates will store session data in a filesystem folder
app.logger.addHandler(file_handler)
Session(app) #create Session instance


lang_id_model = pickle.load(open('lang_id_model.pkl', 'rb'))
lang_sens_model = pickle.load(open('lang_sens_model.pkl', 'rb'))
bow_model_char = pickle.load(open('bow_model_char.pkl', 'rb'))
bow_model = pickle.load(open('bow_model.pkl', 'rb'))
tun_bow_model = pickle.load(open('tun_bow_model.pkl', 'rb'))
tun_lang_sens_model = pickle.load(open('tun_lang_sens_model.pkl', 'rb'))
def allowed_file(filename):
    '''Checks uploaded file to make sure it is.txt'''
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
   return render_template('home.html')

@app.route('/submit', methods=('GET', 'POST'))
def submit():
    form = MyForm()
    if form.validate_on_submit():
        session['file_contents'] = form.name.data
        return redirect('/analysis')
    return render_template('submit.html', form=form)

@app.route('/upload', methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            session['file_contents'] = file.read().decode('utf-8')
            return redirect(url_for('analysis'))
    return render_template('upload.html')

@app.route('/analysis')
def analysis():
    file_contents = session.get('file_contents')
    
    documents = file_contents.splitlines()
    
    wiou=pd.Series(documents)
    dtm=bow_model_char.transform(wiou)
    dtmsens=bow_model.transform(wiou)
    x=len(documents)
    z=range(x,)
    lang_sent = lang_id_model.predict(dtm)
    com_sent = file_contents
    sens_sent =[] 
    
    for n,j in enumerate(lang_sent) :
        if j=='AR':
            print(file_contents[n])
            dtmsens=bow_model.transform(pd.Series(documents[n]))
            if lang_sens_model.predict(dtmsens)[0]==(-1):
                sens_sent.append('Negative -')
            else :
                sens_sent.append('Positive +')
        else:
            dtmsens=tun_bow_model.transform(pd.Series(documents[n]))
            if tun_lang_sens_model.predict(dtmsens)[0]==(-1):
                sens_sent.append('Negative -')
            else :
                sens_sent.append('Positive +')

    return render_template('analysis.html',z=z, lang_sent=lang_sent, sens_sent=sens_sent , com_sent=documents)

@app.route('/about')
def about():
    return render_template('about.html')
if __name__ == '__main__':
   app.run()


wiou=pd.Series(['كوميدي  ','ahla'])
dtm=bow_model_char.transform(wiou)
print(lang_id_model.predict(dtm))