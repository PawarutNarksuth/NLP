from calendar import c
from tabnanny import check
import tempfile
from tkinter.ttk import Style
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

import itertools
import nltk

from nltk.corpus import stopwords
from lib2to3.pgen2 import token
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.tfidfmodel import TfidfModel
from gensim.corpora.dictionary import Dictionary

from gensim.corpora.dictionary import Dictionary
from collections import defaultdict

import spacy
from spacy import displacy

from flaskext.markdown import Markdown

import Model.tagLabel as imModel #เรียก import file python โดยใช้แบบ oop 

app = Flask(__name__)
Markdown(app)

art = []
nameFile = []
tempBowID = []
tempBowWord = []
tempTF = []
articles = []
ckk = True
longfile = 0

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/spacy')
def spa():
    return render_template("page_spacy.html")

@app.route('/model')
def pagemodel():
    return render_template("page_model.html")

@app.route('/upload_model' , methods=['GET','POST'] )
def loadDel():
    text = request.form.getlist('TextA')
    Ans = imModel.get_prediction(text , convert_to_label=True)

    return render_template("page_model.html" , Ans = Ans)

@app.route('/upload_spacy' , methods=['GET','POST'])
def loadspa():
    global ckk
    if(request.method == 'POST'):
        f = request.files.getlist('fileText')
        #f.save(secure_filename(f.filename)

        fn = os.listdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file"))
        for i in fn:
            os.remove(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file" , i))

        for files in f:
            #files.save(secure_filename(files.filename))
            
            files.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file", files.filename))

            temp_file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file" , files.filename) ,"r")

            article = temp_file.read()
            art.append(article)
            nameFile.append(files.filename)

            #ทำให้อยู่ในรูป token
            tokens = word_tokenize(article)

            #ทำให้เป็นตัวพิมพ์เล็ก
            lower_tokens = [t.lower() for t in tokens]

            #ทำให้มีแต่ตัวหนังสือ
            alpha_only = [t for t in lower_tokens if t.isalpha()]

            #เอาคำที่เป็น stopword ออก
            no_stops = [t for t in alpha_only if t not in stopwords.words('english')]

            #ทำ lemmatizer ให้เป็นคำศัพท์จริงๆของมัน
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

            #เอาเข้า list
            articles.append(lemmatized)

        nlp = spacy.load("en_core_web_sm")
        doc = nlp(article)
        
        dis = displacy.render(doc, style="ent")
        display = dis
        print(display)
        
    return render_template("page_spacy.html" ,disP=display ,ckk=ckk)

@app.route('/upload', methods=['GET','POST'])
def loadFile():
    global art , nameFile , tempBowID , tempBowWord ,tempTF ,articles ,longfile
    art = []
    nameFile = []
    tempBowID = []
    tempBowWord = []
    tempTF = []
    articles = []
    ckk = True
    longfile = 0
    
    if(request.method == 'POST'):
        f = request.files.getlist('fileText')
        #f.save(secure_filename(f.filename))
        count = 0
        for i in f:
            count += 1
        longfile = count    
        fn = os.listdir(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file"))
        for i in fn:
            os.remove(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file" , i))

        for files in f:
            #files.save(secure_filename(files.filename))
            
            files.save(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file", files.filename))

            temp_file = open(os.path.join(os.path.abspath(os.path.dirname(__file__)), "All_file" , files.filename) ,"r")

            article = temp_file.read()
            art.append(article)
            nameFile.append(files.filename)

            #ทำให้อยู่ในรูป token
            tokens = word_tokenize(article)

            #ทำให้เป็นตัวพิมพ์เล็ก
            lower_tokens = [t.lower() for t in tokens]

            #ทำให้มีแต่ตัวหนังสือ
            alpha_only = [t for t in lower_tokens if t.isalpha()]

            #เอาคำที่เป็น stopword ออก
            no_stops = [t for t in alpha_only if t not in stopwords.words('english')]

            #ทำ lemmatizer ให้เป็นคำศัพท์จริงๆของมัน
            wordnet_lemmatizer = WordNetLemmatizer()
            lemmatized = [wordnet_lemmatizer.lemmatize(t) for t in no_stops]

            #เอาเข้า list
            articles.append(lemmatized)

        tempBowID = BowID(articles)
        tempBowWord = BowWord(articles)
        tempTF= Tf(articles)

        return render_template("index.html" , tempBowID = tempBowID , tempBowWord = tempBowWord , tempTF = tempTF , art = art , nameF = nameFile ,longfile = longfile)

@app.route('/search', methods=['GET','POST'])
def search():
    global ckk

    dictionary = Dictionary(articles)
    computer_id = dictionary.token2id.get(request.form["srh"])
    
    print(computer_id)

    print(dictionary.get(computer_id))

    if(dictionary.get(computer_id) != None):
        text = "Found your  word : "+ request.form["srh"]
        ckk = True
    else:
        text = "Not found your  word : "+ request.form["srh"]
        ckk = False

    return render_template("index.html" , tempBowID = tempBowID , tempBowWord = tempBowWord , tempTF = tempTF , art = art , nameF = nameFile ,text = text , ckk=ckk ,longfile = longfile)
    
def BowID(articles):
    temp_id = []

    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
        
    total_word_count = defaultdict(int)
    for word_id , word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count

    sorted_word_count = sorted(total_word_count.items() , key=lambda w:w[1] , reverse=True)
    
    for word_id ,word_count in sorted_word_count[0:5]:
        temp_id.append(dictionary.get(word_id))
    
        print(dictionary.get(word_id))
    return temp_id 

def BowWord(articles):
    temp_word = []

    dictionary = Dictionary(articles)
    corpus = [dictionary.doc2bow(a) for a in articles]
        
    total_word_count = defaultdict(int)
    for word_id , word_count in itertools.chain.from_iterable(corpus):
        total_word_count[word_id] += word_count

    sorted_word_count = sorted(total_word_count.items() , key=lambda w:w[1] , reverse=True)
    
    for word_id ,word_count in sorted_word_count[0:5]:
        temp_word.append(word_count)

        print(dictionary.get(word_count))
    return temp_word 

def Tf(articles):
    temp_TF = []

    #สร้าง Dictionary จาก articles 
    dictionary = Dictionary(articles)

    #สร้างตัว corpus
    corpus = [dictionary.doc2bow(a) for a in articles]

    #สร้าง Tf-idf model
    tfidf = TfidfModel(corpus)

    All_tfidf = []

    for inCor in corpus:
        All_tfidf += tfidf[inCor]
    
    #นำตัว tfidf_weight มาเรียง
    sorted_tfidf_weight = sorted(All_tfidf, key=lambda w:w[1] ,reverse=True)

    for term_id , weight in sorted_tfidf_weight[0:5]:
        temp_TF.append(weight)
        print(dictionary.get(term_id) , weight)
    
    return temp_TF

if __name__ == "__main__":
    app.run(debug=True , port=8080)
