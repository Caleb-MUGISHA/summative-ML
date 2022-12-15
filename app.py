import gensim
import gensim.downloader
import numpy as np
import pandas as pd
import tensorflow_hub as hub
from gensim.models import Word2Vec
from nltk.corpus import stopwords
from nltk import download
from pyemd import emd
from sklearn.metrics.pairwise import cosine_similarity
from autocorrect import Speller


import os
import re
import sys
import time

from flask import Flask, redirect, render_template, request, make_response, jsonify
from flask_ngrok import run_with_ngrok
from werkzeug.utils import secure_filename
import os

stopword = download('stopwords')
stop_words = stopwords.words('english')

"""
Takes a list of uploaded filenames and returns list of file_text
"""
def get_ocr_text(uploaded_files):
    to_return = []

    if not uploaded_files:
        print("No files uploaded!")
    
    check = Speller(lang='en')
    
path_to_saved_model = gensim.downloader.load('word2vec-google-news-300', return_path=True)
model = gensim.models.KeyedVectors.load_word2vec_format(path_to_saved_model, binary=True)  
model.init_sims(replace=True)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def cos_sim(input_vectors):
    similarity = cosine_similarity(input_vectors)
    return similarity

negative = ["not" , "without", "against", "bad", "useless", "no", "dislike", "hate"]

def semantic_similarity(actual_answer , given_answer) :
    actual = actual_answer.lower().split(".")
    given = given_answer.lower().split(".")
    
    sim_checker = actual 
    
    not_matching_semantics = list()
    
    semantic_1 = 0   # Actual_answer
    semantic_2 = 0   # Given_answer
    
    actual_embed_list = list()
    given_embed_list = list()
    
    for z in range(len(actual)) :
        list_actual = list()  
        list_actual.append(actual[z])
        actual_embed_list.append(embed(list_actual))
        #print(actual_embed_list[z].shape)
    
    for z in range(len(given)) :    
        semantic_1 = 0
        semantic_2 = 0 
        list_given = list()
        list_given.append(given[z])
        embed_z = embed(list_given)
        
        sim_check = sim_checker.copy() 
        sim_check.append(given[z]) 
        
        sen_em = embed(sim_check)
        
        similarity_matrix = cos_sim(np.array(sen_em))
        
        similarity_matrix_df = pd.DataFrame(similarity_matrix) 
        
        cos_list = list(similarity_matrix_df[len(similarity_matrix_df) - 1]) 
        cos_list = cos_list[:len(cos_list)-1]
        #print(cos_list)
        
        index = cos_list.index(max(cos_list))
        
        actual_check = actual[index]
        actual_check = actual_check.split()
        for i in range(len(actual_check) - 1) :
            if(actual_check[i] in negative and actual_check[i+1] in negative) :
                semantic_1 += 1 
            elif(actual_check[i] in negative and actual_check[i+1] not in negative) :
                semantic_1 -= 1

        answer_given = given[z].split()
        for i in range(len(answer_given) - 1) :
            if(answer_given[i] in negative and answer_given[i+1] in negative) :
                semantic_2 += 1 
            elif(answer_given[i] in negative and answer_given[i+1] not in negative) :
                semantic_2 -= 1 
        
        if(semantic_1 == 0 and semantic_2 == 0) :
            
            """
            Well and good
            """
        elif(semantic_1 < 0  and semantic_2 >= 0) :
            not_matching_semantics.append(list([actual[index],given[z]]))
            embed_z*=(-1)

        elif(semantic_1 >= 0 and semantic_2 < 0 ) :
            not_matching_semantics.append(list([actual[index],given[z]]))
            embed_z*=(-1)
        
        #print(semantic_1,semantic_2,actual[index],given[z])
        given_embed_list.append(embed_z)
    
    #print(np.array(actual_embed_list).shape)
    actual_embed = actual_embed_list[0] 
    #print(actual_embed.shape) 
    
    for i in range(len(actual_embed_list)-1) :
        #print(actual_embed_list[i+1].shape)
        actual_embed += actual_embed_list[i+1]
        
    given_embed = given_embed_list[0] 
    for i in range(len(given_embed_list) - 1) :
        given_embed += given_embed_list[i+1] 
            
    actual_embed = np.array(actual_embed).reshape(512)
    given_embed = np.array(given_embed).reshape(512) 
    sem_checker = list([actual_embed,given_embed]) 
    answer = pd.DataFrame(cos_sim(sem_checker))
        
    return not_matching_semantics , answer[0][1]

def WMD(actual_answer, given_answer, model) :
    actual_answer = actual_answer.lower().split()
    actual_answer = [w for w in actual_answer if w not in stop_words]
    
    given_answer = given_answer.lower().split()
    given_answer = [w for w in given_answer if w not in stop_words]
    
    return model.wmdistance(given_answer,actual_answer)

def score(given_answer, actual_answer, model) :
    given_answer1 = given_answer[:]
    actual_answer1 = actual_answer[:]
    
    given_answer2 = given_answer[:]
    actual_answer2 = actual_answer[:]

    not_matching , similarity = semantic_similarity(actual_answer1, given_answer1)
    distance = WMD(actual_answer2, given_answer2, model)
    
    # if(similarity > 0) :
    # if(distance == 0) :
    #     return 1 
    print("NOT MATCHING TEXT: ", not_matching)
    return similarity/distance
    # else :
        # return -1

app = Flask(__name__)

UPLOAD_FOLDER = os.path.join("static","assets","img", 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['JSON_SORT_KEYS'] = False

def file_name(filename):
    return os.path.join(UPLOAD_FOLDER, filename)

@app.route("/")
@app.route("/index")
def home():
    return render_template("index.html")

@app.route("/live-demo")
def live_demo():
    return render_template("live-demo.html")

@app.route("/result", methods = ['GET', 'POST'])
def result():
    print("******* Inside RESULT **********")
    if request.method == 'POST':
        print("SUMMATIVE: ", request.files)

        teacher_file = request.files.get('file1', None)
        student_files = request.files.getlist('file2', None)

        # print("Got files: ", file1, file2)

        if teacher_file and student_files:
            teacher_file.save(file_name(teacher_file.filename))

            for file_ in student_files:
                file_.save(file_name(file_.filename))

            teacher_text = [file_name(teacher_file.filename)][0]
            students_texts = [file_name(f.filename) for f in student_files]
            
            resulting_scores = [score(teacher_text, stud, model) for stud in students_texts]            
            print("res is of length", len(resulting_scores))

            # return redirect(url_for(""))
            results_dict = {
                "Teacher Filepath": file_name(teacher_file.filename),
                "Teacher Filename": teacher_file.filename,
                "Student Grades": [{
                    "Filepath": file_name(f.filename),
                    "Filename": f.filename,
                    "Score": f"{r:.4f}",
                } for f, r in zip(student_files, resulting_scores)],
            }
            return render_template("output.html", result=results_dict)
    return make_response("Invalid somehow!")#redirect(url_for('home'))

# @app.route("/results")
def output():
    pass

app.run()
