import numpy as np
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_csv("datasets_materials_v29.csv")
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.after_request
def add_header(response):
    response.cache_control.max_age = 100
    return response


@app.route("/property")
def features_csv():
    features = ['keyword','texture','application','transform']
    # features = ['keyword','propertyB','propertyC','propertyD','texture','transform','application']
    print(df.columns)

    def combine_features(row):
        return row['keyword']+" "+row['texture']+" "+row['application']+" "+row['transform']

    for feature in features:
        df[feature] = df[feature].fillna('')

    df["combined_features"] = df.apply(combine_features,axis=1)

    df.iloc[0].combined_features

    # tokenizer
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df["combined_features"])

    #cosine similarity
    cosine_sim = cosine_similarity(count_matrix)


    def get_title_from_index(index):
        return df[df.index == index]["materialA"].values[0]

    #get link for reference
    def get_home(index):
        return df[df.index == index]["link"].values[0]

    def get_index_from_title(title):
        return df[df.application == title]["index"].values[0]

    def get_materialB(index):
        return df[df.index == index]["materialB"].values[0]

    def get_materialC(index):
        return df[df.index == index]["materialC"].values[0]

    def get_materialD(index):
        return df[df.index == index]["materialD"].values[0]

    def get_transform(index):
        return df[df.index == index]["transform"].values[0]

    def get_texture(index):
        return df[df.index == index]["texture"].values[0]

    def get_application(index):
        return df[df.index == index]["application"].values[0]


    picked_title = request.args['chooseProperty']
    project_index = get_index_from_title(picked_title)
    similar_project = list(enumerate(cosine_sim[project_index])) #similarity
    print(similar_project)

    sorted_similar_projects = sorted(similar_project, key=lambda x: x[1], reverse=True)[1:]
    print(sorted_similar_projects)

    i=0
    print("\nPicked Function: "+picked_title)
    print("\nPossible combinations of materials to {"+picked_title+"} are:\nMaterial A | Material B | Material C | Material D => Transform [Property]\n")
    for element in sorted_similar_projects:
        property= "|| "+get_title_from_index(element[0])+" | "+ get_materialB(element[0])+" | "+get_materialC(element[0])+" | "+get_materialD(element[0])+"|| => "+get_transform(element[0])+"  ["+get_application(element[0])+"]"
        i=i+1
        if i>5:
            break
    return render_template('result.html',  property=property)

if __name__ == '__main__':
    app.run()
