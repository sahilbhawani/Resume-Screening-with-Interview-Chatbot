import threading
import random
import pandas as pd
import pdfkit

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from django.http import StreamingHttpResponse, HttpResponse
from django.shortcuts import render
import cv2
from django.contrib import messages
from django.views.decorators import gzip
import warnings
import docx2txt
from pdf2docx import Converter
from .models import Student, Questions
import textdistance as td
import csv
from django.http import FileResponse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk import sent_tokenize
import numpy as np
warnings.filterwarnings('ignore')
face_cascade = cv2.CascadeClassifier('media/cascade_filters/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('media/cascade_filters/haarcascade_eye.xml')


# Create your views here.

def index(request):
    return render(request, 'index.html')


def registration_page(request):
    return render(request, 'registration.html')


def register(request):
    name = request.POST.get('student_name')
    email = request.POST.get('email')
    college = request.POST.get('college')
    cgpa = request.POST.get('cgpa')
    resume = request.FILES.get('resume')

    student = Student()
    student.student_name = name
    student.student_email = email
    student.student_college = college
    student.student_cgpa = cgpa
    student.student_resume = resume
    student.save()
    student1 = Student.objects.get(student_email=email)


    resume_result = resume_screening(student1.student_resume, student1.student_email)
    job_description = docx2txt.process('media/content/sample_description.docx')
    resume = docx2txt.process('media/resumes/docx/' + email + '_resume.docx')
    r1 = process_tfidf_similarity(job_description, resume)
    r1 = '{0:.2f}'.format(r1 * 100)
    # r2 = process_use_similarity(job_description, resume)
    # r2 = '{0:.2f}'.format(r2 * 100)

    r3 = process_bert_similarity(job_description, resume)
    r3 = '{0:.2f}'.format(r3 * 100)
    r4 = Jaccard_similarity(job_description,resume)
    r4 = '{0:.2f}'.format(r4*100)

    print('Resume Matches by: ' + resume_result + '%')
    student1.student_resume_result = resume_result
    context = {'student': student1,'tfidf': str(r1)+'%',"bert":str(r3)+"%","jac":str(r4)+"%" }
    return render(request, 'interview.html', context)

def process_bert_similarity(base_document,documents):
    # print("Base : ",base_document)
    # print(documents)
    # print("$%")
   # This will download and load the pretrained model offered by UKPLab.
   model = SentenceTransformer('bert-base-nli-mean-tokens')

   print("Base : ",base_document)

   print(documents)
   print("$%")
   #print(documents)
   # Although it is not explicitly stated in the official document of sentence transformer, the original BERT is meant for a shorter sentence. We will feed the model by sentences instead of the whole documents.
   sentences = sent_tokenize(base_document)
   base_embeddings_sentences = model.encode(sentences)
   base_embeddings = np.mean(np.array(base_embeddings_sentences), axis=0)
   documents = [documents]
   vectors = []
   for i, document in enumerate(documents):
      print("In Llp")
      print(document)

      sentences = sent_tokenize(document)
      embeddings_sentences = model.encode(sentences)
      embeddings = np.mean(np.array(embeddings_sentences), axis=0)

      vectors.append(embeddings)

      #print("making vector at index:", i)

   scores = cosine_similarity([base_embeddings], vectors).flatten()

   highest_score = 0
   highest_score_index = 0
   for i, score in enumerate(scores):
      if highest_score < score:
         highest_score = score
         highest_score_index = i

   most_similar_document = documents[highest_score_index]
   return highest_score

#Use Similarity
def process_use_similarity(base_docuement,documents):
    filename = "/universal-sentence-encoder_4"
    model = embed
    base_embeddings = model([base_docuement])
    embeddings = model([documents])
    scores = cosine_similarity(base_embeddings, embeddings).flatten()
    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(scores):
        if highest_score < score:
            highest_score = score
            highest_score_index = i
    most_similar_document = documents[highest_score_index]
    print("Most similar document by USE with the score:", highest_score)
    return highest_score

#JACCARD
def Jaccard_similarity(r,d):
    # Split the documents and create tokens
    r = set(r.lower().split())
    d = set(d.lower().split())

    # Calculate the Jaccard Similarity
    jaccard_distance = len(r.intersection(r))/len(r.union(d))

    # Print the Jaccard Simialrity score
    print(jaccard_distance)
    return 1-jaccard_distance


def process_tfidf_similarity(base_document, documents):
    vectorizer = TfidfVectorizer()
    documents = [documents]

    # To make uniformed vectors, both documents need to be combined first.
    documents.insert(0, base_document)
    embeddings = vectorizer.fit_transform(documents)

    cosine_similarities = cosine_similarity(embeddings[0:1], embeddings[1:]).flatten()

    highest_score = 0
    highest_score_index = 0
    for i, score in enumerate(cosine_similarities):
        if highest_score < score:
            highest_score = score
            highest_score_index = i

    most_similar_document = documents[highest_score_index]
    return highest_score


def resume_screening(path, email):
    docx_file = 'media/resumes/docx/' + email + '_resume.docx'
    cv = Converter(path)
    cv.convert(docx_file)
    cv.close()
    job_description = docx2txt.process('media/content/sample_description.docx')
    resume = docx2txt.process('media/resumes/docx/' + email + '_resume.docx')
    content = [job_description, resume]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(content)
    mat = cosine_similarity(count_matrix)
    result = '{0:.2f}'.format(mat[1][0] * 100) + '%'
    return result


@gzip.gzip_page
def video_feed(request):
    try:
        cam = VideoCapture()
        error = cam.error
        print('error', cam.error)
        messages.error(request, error)
        return StreamingHttpResponse(gen(cam), content_type="multipart/x-mixed-replace;boundary=frame")
    except:
        pass
    return render(request, 'interview.html')


def interview(request, student_email):
    interview.a = 0
    print(student_email)
    student = Student.objects.get(student_email=student_email)
    questions_list, questions = generateQuestions()
    print(questions)
    for i in range(0, len(questions)):
        question = Questions()
        question.question = list(questions.keys())[i]
        question.answer = list(questions.values())[i]
        question.student_id = student
        question.save()
    context = {'student': student, 'questions': questions, "questions_list": questions_list}
    print(messages.get_messages(request=request))
    return render(request, 'video.html', context)

class VideoCapture(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        (self.grabbed, self.frame) = self.video.read()
        threading.Thread(target=self.update, args=()).start()
        self.error = ""

    def __del__(self):
        self.video.release()

    def get_frame(self):
        img = self.frame
        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

    def update(self):
        classNames = []
        classFile = 'media/content/classes.names'

        with open(classFile, 'rt') as f:
            classNames = f.read().rstrip('\n').split('\n')

        configPath = 'media/content/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
        weightPath = 'media/content/frozen_inference_graph.pb'

        net = cv2.dnn_DetectionModel(weightPath, configPath)
        net.setInputSize(320, 230)
        net.setInputScale(1.0 / 127.5)
        net.setInputMean((127.5, 127.5, 127.5))
        net.setInputSwapRB(True)
        while True:
            (self.grabbed, self.frame) = self.video.read()
            gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.3,
                minNeighbors=5,
                minSize=(30, 30)
            )
            classIds, confs, bbox = net.detect(self.frame, confThreshold=0.5)
            # confThreshold
            if len(classIds) != 0:
                for classid, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    cv2.rectangle(self.frame, box, color=(0, 255, 0), thickness=1)
                    cv2.putText(self.frame, classNames[classid - 1], (box[0] + 10, box[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (0, 255, 0), thickness=1)
            print(len(faces))
            if len(faces)>= 1 or len(faces)>=2:
                self.error += "Mobile phone or another person detected in window"
                for classid, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                    if classNames[classid - 1] == "cell phone":
                        self.error += "Mobile phone or another person detected in window"
                        # print("Mobile phone or another person detected in window")
            if (interview.a==1):
                break


def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def generateQuestions():
    data = pd.read_csv('media/content/Question_Answer.csv')
    questions = {}
    r = random.randint(0, 430)
    for i in range(r, r + 8):
        questions[data.iloc[i]['Q']] = data.iloc[i]['A']
    return list(questions.keys()), questions


def result(request, student_id):
    interview.a=1
    student_answers = []
    for key, value in request.POST.items():
        student_answers.append(value)
    student_answers.pop(0)
    student = Student.objects.get(student_id=student_id)
    result1 = []
    questions = list(Questions.objects.filter(student_id=student_id))
    for i in range(0, len(student_answers)):
        result1.append(td.sorensen.normalized_similarity(questions[i].answer, student_answers[i]) * 100)
    header = ["Student Id", "Student Name", "Question", "Original Answer", "Student Answer", "Result"]
    print(header)
    data = []
    for i in range(0, len(student_answers)):
        list1 = [student.student_id, student.student_name, questions[i].question, questions[i].answer,
                 student_answers[i], str(round(result1[i], 2))]
        data.append(list1)
    print(data)
    result2 = 0
    for i in range(0, len(student_answers)):
        result2 += result1[i]
    print(result2)
    result2 = (result2/len(student_answers))
    student.student_result = result2
    student.save()
    footer = [student.student_id,student.student_name, " ", " ", "Total", str(round(result2, 2)) + "%"]
    print(footer)
    with open('media/result/csv/' + str(student_id) + '.csv', 'w', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)
        writer.writerow(footer)
    df1 = pd.read_csv("media/result/csv/" + str(student_id) + ".csv")
    html_string = df1.to_html()
    config = pdfkit.configuration(wkhtmltopdf=bytes(r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe', 'utf8'))
    options = {
        'page-size': 'A4',
        'margin-top': '0.75in',
        'margin-right': '0.75in',
        'margin-bottom': '0.75in',
        'margin-left': '0.75in',
        'encoding': "UTF-8",
        'custom-header': [
            ('Accept-Encoding', 'gzip')
        ],
        'no-outline': None
    }
    pdfkit.from_string(html_string, "media/result/pdf/" + str(student_id) + ".pdf", configuration=config,
                       options=options)
    context = {'student_id': student_id}
    return render(request, 'result.html', context)


def download(request, student_id):
    return FileResponse(open('media/result/pdf/' + str(student_id) + ".pdf", 'rb'), as_attachment=True,
                        content_type='application/pdf')