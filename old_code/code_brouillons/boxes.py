import cv2
import face_recognition
import pickle
import os 
os.chdir("C:\\Users\\rpic\\Documents\\PHYTEM\\2020-2021\\Cours\\WebMining\\Challenge_SISE")
mypath="C:\\Users\\rpic\\Documents\\PHYTEM\\2020-2021\\Cours\\WebMining\\Challenge_SISE/training"

data = pickle.loads(open('face_enc', "rb").read())
#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
face_code={}
for img in onlyfiles[0:1] :
    
    picture = cv2.imread(mypath+'/'+img)
    wb = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(50, 50),
                                         flags=cv2.CASCADE_SCALE_IMAGE)
    boxes = face_recognition.face_locations(wb,model='hog')
    print(boxes,faces)


