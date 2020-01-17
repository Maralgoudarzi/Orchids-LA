import numpy as np
from flask import Flask, render_template, flash, request, url_for
from werkzeug.utils import redirect
from Model_classification import predictFunction, validation_data
from Convertor import *
import plot as plt

# App config.
DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config['SECRET_KEY'] = 'M123A456R678A90L'

studentInfo = ["gender","NationalITy","PlaceofBirth","StageID","GradeID","SectionID","Topic","Semester",
               "Relation","","raisedhands","VisITedResources","AnnouncementsView","Discussion","ParentAnsweringSurvey","ParentschoolSatisfaction","StudentAbsenceDays"]

# Route for handling the login page logic
@app.route('/')
def main():
    return render_template('index.html')

@app.route('/Students', methods=['GET', 'POST'])
def dataForm():
    error: None
    gender : ['gender', 'M' , 'F']
    Nationality : ['select your Nationality', 'KW',' USA', 'Jordan', 'Iran', 'lebanon', 'SaudiArabia', 'Egypt',' Tunis',' Morocco',' Syria',' Lybia','Palestine','Iraq']
    PlaceofBirth : ['select your place of Birth', 'KuwaIT','USA','Jordan','Iran','lebanon','SaudiArabia','Egypt','Tunis','Morocco','Syria','Lybia','Palestine','Iraq']
    StageID : ['select your stage ID', 'lowerlevel',' MiddleSchool','HighSchool']
    GradeID : ['select your stage ID','G-01','G-02','G-03','G-04','G-05','G-06','G-07','G-08','G-09','G-10','G-11','G-12']
    SectionID : ['your Section ID','A','B','C']
    Topic : ['which Topic','IT','Math','Arabic','Science','English','Quran','Spanish','French','Arabic','History','Biology','Geology','Chemistry']
    Semester : ['Which semester','S','F']
    Relation : ['Your Relation with StudentFather','Father','Mum']
    raisedhands : ['How many times studens raise hand', 0 : 100]
    VisITedResources : ['How many times does student attend at course',0:100]
    AnnouncementsView : ['How many times does student view the lectures',0:100]
    Discussion : ['does student take part in Discussion',0:100]
    ParentAnsweringSurvey : ['ParentAnsweringSurvey','Yes','No']
    StudentAbsenceDays : ['How many times was student absent ','Under-7','Above-7']


    if request.method == 'POST':
        if request.form['gender'] == 'select your gender' \
           or request.form['NationalITy'] == 'select your Nationality' \
           or request.form['PlaceofBirth'] == 'select your place of Birth' \
           or request.form['StageID'] == 'select your stage ID' \
           or request.form['GradeID'] == 'give your Grade' \
           or request.form['SectionID'] == 'your Section ID' \
           or request.form['Topic'] == 'which Topic' \
           or request.form['Semester'] == 'Which semester' \
           or request.form['Relation'] == 'Your Relation with Student' \
           or request.form['raisedhands'] == 'How many times studens raise hand' \
           or request.form['VisITedResources'] == 'How many times does student attend at course' \
           or request.form['AnnouncementsView'] == 'How many times does student view the lectures' \
           or request.form['Discussion'] == 'does student take part in Discussion' \
           or request.form['ParentAnsweringSurvey'] == 'ParentAnsweringSurvey' \
           or request.form['ParentschoolSatisfaction'] == 'ParentschoolSatisfaction' \
           or request.form['StudentAbsenceDays'] == 'How many times was student absent'
                error : "select all field."
        else:
            studentInfo[0] = convertor(request.form['gender'])
            studentInfo[1] = convertor(request.form['NationalITy'])
            studentInfo[2] = convertor(request.form['PlaceofBirth'])
            studentInfo[3] = convertor(request.form['StageID'])
            studentInfo[4] = convertor(request.form['GradeID'])
            studentInfo[5] = convertor(request.form['SectionID'])
            studentInfo[6] = convertor(request.form['Topic'])
            studentInfo[7] = convertor(request.form['Semester'])
            studentInfo[8] = convertor(request.form['Relation'])
            studentInfo[9] = convertor(request.form['raisedhands'])
            studentInfo[10] = convertor(request.form['VisITedResources'])
            studentInfo[11] = convertor(request.form['AnnouncementsView'])
            studentInfo[12] = convertor(request.form['Discussion'])
            studentInfo[13] = convertor(request.form['ParentAnsweringSurvey'])
            studentInfo[14] = convertor(request.form['ParentschoolSatisfaction'])
            studentInfo[15] = convertor(request.form['StudentAbsenceDays'])

            return redirect (url_for ('Prediction'))
            return render_template('Students.html', error=error,gender=gender,NationalITy =NationalITy,PlaceofBirth=PlaceofBirth, StageID=StageID,
                           GradeID=GradeID,SectionID=SectionID,Topic=Topic,Semester=Semester,Relation=Relation,raisedhands=raisedhands,
VisITedResources=VisITedResources,AnnouncementsView=AnnouncementsView,Discussion=Discussion,ParentAnsweringSurvey=ParentAnsweringSurvey,ParentschoolSatisfaction=ParentschoolSatisfaction,StudentAbsenceDays=StudentAbsenceDays)

@app.route('/prediction')
def prediction():
    df['TotalQ'] = df['Class']
    df['TotalQ'].loc[df.TotalQ == 'Low-Level'] = 0.0
    df['TotalQ'].loc[df.TotalQ == 'Middle-Level'] = 1.0
    df['TotalQ'].loc[df.TotalQ == 'High-Level'] = 2.0

    continuous_subset = df.iloc[:, 9:13]
    continuous_subset['gender'] = np.where(df['gender'] == 'M', 1, 0)
    continuous_subset['Parent'] = np.where(df['Relation'] == 'Father', 1, 0)

    X = np.array(continuous_subset).astype('float64')
    y = np.array(df['TotalQ'])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)

    knn: predict = KNeighborsClassifier(n_neighbors=23)
    knn.fit(X_train, y_train)

    return render_template('prediction.html',knn)

if __name__ == "__main__":
    app.run()