#----------------------------------------------------------------------#
#                         IMPORT DES PACKAGES                          #-------------------------------------------------------
#----------------------------------------------------------------------#

import pandas as pd
import face_recognition
import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time
import math
import pickle
import argparse
import numpy as np
import glob
from tkinter import *
from collections import Counter
from tkinter import messagebox
from tkinter import filedialog
from tkinter import ttk
import PIL.Image, PIL.ImageTk
from shutil import copyfile
import os
from pandastable import Table, TableModel
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.chdir("C:/GITHUB/Reconnaissance_faciale_challenge_SISE")
#----------------------------------------------------------------------#
#          DEFINITION D'UNE FONCTION QUI RETROUVE UN VISAGE            #-------------------------------------------------------
#----------------------------------------------------------------------#

def highlightFace(net, frame, conf_threshold=0.55):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.25, (350, 350), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes



#----------------------------------------------------------------------#
#          DEFINITION D'UNE FONCTION QUI CREER UN RESUME               #-------------------------------------------------------
#----------------------------------------------------------------------#

def append_frame_info(lst,name,gender,age,angry,disgusted,feareful,happy,neutral,sad,surprised):
    lst.append([name,gender,age,angry,disgusted,feareful,happy,neutral,sad,surprised])
    return lst

def create_resume(lst):
    person = []
    age = []
    genre = []
    angry =[]
    disgusted =[]
    feareful =[]
    happy =[]
    neutral =[]
    sad =[]
    surprised =[]
    
    # récupération des personnes, âges et des sexes dans des listes
    for i in range(len(lst)):
        person.append(lst[i][0])
        genre.append(lst[i][1]) 
        age.append(lst[i][2])
        angry.append(lst[i][3])
        disgusted.append(lst[i][4])
        feareful.append(lst[i][5])
        happy.append(lst[i][6])
        neutral.append(lst[i][7])
        sad.append(lst[i][8])
        surprised.append(lst[i][9])
        
        
    d = {'person' : person, 'genre' : genre, 'age' : age,'angry':angry,'disgusted':disgusted,'feareful':feareful,'happy':happy,'neutral':neutral,'sad':sad,'surprised':surprised }
    donnees = pd.DataFrame(data=d)
    
    
    resum = []
    for persons in np.unique(person):
        df_mask=donnees['person']==persons
        filtered_df = donnees[df_mask]
        name = np.unique(filtered_df.iloc[:,0])
        age_moy = round(np.mean(filtered_df.iloc[:,2]),0)
        angry_moy= round(np.mean(filtered_df.iloc[:,3]),2)
        disgusted_moy= round(np.mean(filtered_df.iloc[:,4]),2)
        feareful_moy= round(np.mean(filtered_df.iloc[:,5]),2)
        happy_moy= round(np.mean(filtered_df.iloc[:,6]),2)
        neutral_moy= round(np.mean(filtered_df.iloc[:,7]),2)
        sad_moy= round(np.mean(filtered_df.iloc[:,8]),2)
        surprised_moy= round(np.mean(filtered_df.iloc[:,9]),2)
  
        # tableau de la répartiton du sexe
        table_sexe = np.array(Counter(filtered_df.iloc[:,1]).most_common())
        print(table_sexe)
        # recherche du sexe maximum
        maximum = int(table_sexe[0][1])
        genre_moy = table_sexe[0][0]
        
        
        if (len(table_sexe)>1):
            if(int(table_sexe[1][1])>maximum):
                genre_moy = table_sexe[1][0]


        resum.append([name,genre_moy,age_moy,angry_moy,disgusted_moy,feareful_moy,happy_moy,neutral_moy,sad_moy,surprised_moy])
        
    
    resum = pd.DataFrame(resum)
    resum.columns = ['person', 'genre', 'age','angry','disgusted','feareful','happy','neutral','sad','surprised']
    return resum

def controller(img, brightness=255, contrast=127,settings=True): 
        brightness = int((brightness - 0) * (255 - (-255)) / (510 - 0) + (-255)) 
        contrast = int((contrast - 0) * (127 - (-127)) / (254 - 0) + (-127)) 
        if brightness != 0: 
            if brightness > 0: 
                shadow = brightness 
                max = 255
            else: 
                shadow = 0
                max = 255 + brightness 
            al_pha = (max - shadow) / 255
            ga_mma = shadow 
            # The function addWeighted  
            # calculates the weighted sum  
            # of two arrays 
            cal = cv2.addWeighted(img, al_pha, img, 0, ga_mma) 
        else: 
            cal = img 
        if contrast != 0: 
            Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast)) 
            Gamma = 127 * (1 - Alpha) 
      
            # The function addWeighted calculates 
            # the weighted sum of two arrays 
            cal = cv2.addWeighted(cal, Alpha, cal, 0, Gamma) 
      
        # putText renders the specified 
        # text string in the image. 
        if settings:
            cv2.putText(cal, 'B:{},C:{}'.format(brightness, contrast), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2) 
      
        return cal 

#----------------------------------------------------------------------#
#           PARAMETRAGES DES ALGORITHMES : AGE ET GENRE                #-------------------------------------------------------
#----------------------------------------------------------------------#

faceProto="opencv_face_detector.pbtxt"
faceModel="opencv_face_detector_uint8.pb"
ageProto="age_deploy.prototxt"
ageModel="age_net.caffemodel"
genderProto="gender_deploy.prototxt"
genderModel="gender_net.caffemodel"

MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
#ageList=['(0-10)', '(12-14)', '(16-20)', '(21-25)', '(26-35)', '(38-43)', '(48-53)', '(60-80)']
genderList=['Homme','Femme']

faceNet=cv2.dnn.readNet(faceModel,faceProto)
#ageNet=cv2.dnn.readNet(ageModel,ageProto)
ageNet = cv2.dnn.readNetFromCaffe("age.prototxt", "dex_chalearn_iccv2015.caffemodel")
genderNet=cv2.dnn.readNet(genderModel,genderProto)

video = cv2.VideoCapture("video_input/VIDEO1.mp4")
padding=20

#----------------------------------------------------------------------#
#           PARAMETRAGES DES ALGORITHMES : SENTIMENTS                  #-------------------------------------------------------
#----------------------------------------------------------------------#

def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    fig, axs = plt.subplots(1,2,figsize=(15,5))
    # summarize history for accuracy
    axs[0].plot(range(1,len(model_history.history['accuracy'])+1),model_history.history['accuracy'])
    axs[0].plot(range(1,len(model_history.history['val_accuracy'])+1),model_history.history['val_accuracy'])
    axs[0].set_title('Model Accuracy')
    axs[0].set_ylabel('Accuracy')
    axs[0].set_xlabel('Epoch')
    axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/10)
    axs[0].legend(['train', 'val'], loc='best')
    # summarize history for loss
    axs[1].plot(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].plot(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epoch')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['train', 'val'], loc='best')
    fig.savefig('plot.png')
    plt.show()


# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


#----------------------------------------------------------------------#
#           PARAMETRAGES DES ALGORITHMES : VISAGE                      #-------------------------------------------------------
#----------------------------------------------------------------------#

data = pickle.loads(open('face_enc', "rb").read())
#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)

#output_movie = cv2.VideoWriter("C:/GITHUB/easy_facial_recognition-master/easy_facial_recognition-master/Challenge_SISE_Romain/Challenge_SISE/video_output")#video_name.split('_')[0]+'_output/'+video_name.split('/')[1].split('.')[0]+'_output.avi', fourcc, fps, (width, height) )





  

#----------------------------------------------------------------------#
#                   CLASSE APPLICATION PRINCIPALE                      #-------------------------------------------------------
#----------------------------------------------------------------------#


class App:
    def __init__(self, window, window_title, video_source="video_input/VIDEO1.mp4"):
        self.window = window
        self.window.title(window_title)
        self.btn_select=tkinter.Button(window, text="Select video file", width=15, command=self.open_file)
        self.btn_select.pack(anchor=tkinter.NW, expand=True)
        self.video_source = video_source
        self.lst=[]     
        self.f = tkinter.Frame(self.window)
        self.f.pack(anchor=tkinter.CENTER,expand=False)
        self.f.config(width=200, height=200)
        self.f.pack()
        self.f.update_idletasks()
        #pt
        
        # open video source (by default this will try to open the computer webcam)
        #self.vid = MyVideoCapture("C:/GITHUB/easy_facial_recognition-master/easy_facial_recognition-master/PHOTOS+VIDEOS_SISE-20210310T090923Z-001/PHOTOS+VIDEOS_SISE/VIDEO1.mp4")
        self.vid = MyVideoCapture(self.video_source)
        length = int(self.vid.vid.get(cv2.CAP_PROP_FRAME_COUNT))

        

 
        cv2.namedWindow('Frame',cv2.WINDOW_KEEPRATIO&cv2.WINDOW_NORMAL)
        cv2.namedWindow('Parameters',cv2.WINDOW_AUTOSIZE&cv2.WINDOW_FREERATIO)
        #cv2.resizeWindow('Parameters', 400,250)
        
        cv2.createTrackbar( 'start', "Parameters", 0, length, self.vid.onChange)
        cv2.createTrackbar( 'end'  , "Parameters", length, length, self.vid.onChange)
        cv2.createTrackbar('Brightness', 'Parameters', int(255/5), int(255*2/5), self.vid.BrightnessContrast)
        cv2.createTrackbar('Contrast', 'Parameters', int(127/5), int(127*2/5), self.vid.BrightnessContrast)
        
        
        self.vid.onChange(0)
        self.vid.BrightnessContrast(0)
        cv2.waitKey()
        
        self.start = cv2.getTrackbarPos('start',"Parameters")
        self.end   = cv2.getTrackbarPos('end',"Parameters")
        
        self.brightness = 5* cv2.getTrackbarPos('Brightness', 'Parameters') 
        self.contrast = 5* cv2.getTrackbarPos('Contrast', 'Parameters')
        
        if self.start >= self.end:
            raise Exception("start must be less than end")
        

        self.vid.vid.set(cv2.CAP_PROP_POS_FRAMES,self.start)
        cv2.destroyAllWindows()
        #cv2.namedWindow('Process',cv2.WINDOW_KEEPRATIO&cv2.WINDOW_NORMAL)
        self.fps = self.vid.vid.get(cv2.CAP_PROP_FPS)
        self.width  = int(self.vid.vid.get(cv2.CAP_PROP_FRAME_WIDTH) )
        self.height = int(self.vid.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        #output_movie = cv2.VideoWriter(video_name.split('_')[0]+'_output/'+video_name.split('/')[1].split('.')[0]+'_output.avi', fourcc, fps, (width, height) )
        self.output_movie = cv2.VideoWriter('video_output/video_output.avi', self.fourcc, self.fps, (self.width, self.height) )
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.k=0
        self.update()

        self.window.mainloop()
        
    def open_file(self):

        self.pause = False

        self.filename = filedialog.askopenfilename(title="Select file", filetypes=(("MP4 files", "*.mp4"),
                                                                                   ("AVI files", "*.avi")))
        print("\n Filename : ", self.filename, "\n")

        # Open the video file
        self.vid = MyVideoCapture(self.filename)
        self.lst=[]
        # Get video source width and height
        self.canvas = tkinter.Canvas(tkinter.window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack(anchor=tkinter.CENTER)
        #self.update()

        
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()

        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    def update(self):
        
        # Get a frame from the video source
        if (self.vid.vid.get(cv2.CAP_PROP_POS_FRAMES)> self.end):
            frame=None
        else:
            ret, frame = self.vid.get_frame()
            frame= controller(frame,self.brightness,self.contrast, settings=False)
        

        
        if(frame is None):
            self.output_movie.release()
            self.canvas.delete(self.im_canvas)
            res=create_resume(self.lst)
            self.table = Table(self.f, dataframe=res).show()
            

            
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultImg,faceBoxes=highlightFace(faceNet,frame)

        faces = face_recognition.face_locations(gray,model='hog')
        # the facial embeddings for face in input
        
        names = []

        if not faceBoxes:
            print("No face detected")
        else:
            
            self.k+=1
            
            cpt=0
            for faceBox in faces:

                face=frame[max(faceBox[0]-padding,0):min(faceBox[2]+padding,frame.shape[0]),max(0,faceBox[3]-padding):min(faceBox[1]+padding,frame.shape[1])]
        

                    
                    
               # print(face)
                encodings = face_recognition.face_encodings(rgb,[faceBox])
                matches = face_recognition.compare_faces(data["encodings"],encodings[0],tolerance=.5)
                #set name =inknown if no encoding matches
                name = "Unknown"
                # check to see if we have found a match
                if True in matches:
                    #Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        #Check the names at respective indexes we stored in matchedIdxs
                        name = data["names"][i]
                        #increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                    #set name which has highest count
                    name = max(counts, key=counts.get)
                # update the list of names
                names.append(name)
                #---------------------------------- GENRE ----------------------------------   
                #cpt_face=cpt_face+1
                blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                genderNet.setInput(blob)
                genderPreds=genderNet.forward()
                gender=genderList[genderPreds[0].argmax()]
                #print(f'Gender: {gender}')
        
                
                detected_face = cv2.resize(face, (224, 224)) #img shape is (224, 224, 3) now
                img_blob = cv2.dnn.blobFromImage(detected_face ) # img_blob shape is (1, 3, 224, 224)
                
                
                
                #---------------------------------- AGE ----------------------------------  
                
                #prediction
                ageNet.setInput(img_blob)
                age_dist = ageNet.forward()[0]
        
                output_indexes = np.array([i for i in range(0, 101)])
                apparent_predictions = round(np.sum(age_dist * output_indexes), 0)
                
                #---------------------------------- SENTIMENT ----------------------------------  
                
                
                
                # roi_gray = gray[max(0,faceBox[1]-padding):
                #            min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                #            :min(faceBox[2]+padding, frame.shape[1]-1)]
                roi_gray=gray[max(faceBox[0]-padding,0):min(faceBox[2]+padding,gray.shape[0]),max(0,faceBox[3]-padding):min(faceBox[1]+padding,gray.shape[1])]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
                prediction = model.predict(cropped_img)
                maxindex = int(np.argmax(prediction))
                
                        
                
                
                angry= round(prediction[0][0],0)
                disgusted= round(prediction[0][1],0)
                feareful= round(prediction[0][2],0)
                happy= round(prediction[0][3],0)
                neutral= round(prediction[0][4],0)
                sad= round(prediction[0][5],0)
                surprised= round(prediction[0][6],0)
                
                self.lst=append_frame_info(self.lst, name, gender, apparent_predictions,angry,disgusted,feareful,happy,neutral,sad,surprised)
                print(self.lst)
                cv2.rectangle(resultImg,(faceBox[1],faceBox[0]),(faceBox[3],faceBox[2]),(0,255,0),2)
                cv2.putText(resultImg, f'{name}, {gender}, {apparent_predictions}, {emotion_dict[maxindex]}', (faceBox[3], faceBox[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
                
                
                cpt+=1
                if (cpt==len(faces)):
                    if ret:
                        resultImg=cv2.resize(resultImg,(913,515))
                        self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(resultImg))
                        self.im_canvas=self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
                        self.output_movie.write(resultImg)

            
        self.window.after(self.delay, self.update)
    
#----------------------------------------------------------------------#
#                   CLASSE DEFINITION DE LA VIDEO                      #-------------------------------------------------------
#----------------------------------------------------------------------#

class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)



            
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
            

    def onChange(self,trackbarValue):
        start = cv2.getTrackbarPos('start',"Parameters")
        self.vid.set(cv2.CAP_PROP_POS_FRAMES,start)
        err,img = self.vid.read()
        cv2.imshow("Frame", img)
        
        pass
    
    def BrightnessContrast(self,brightness): 
      
        # getTrackbarPos returns the  
        # current position of the specified trackbar.
        start = cv2.getTrackbarPos('start',"Parameters")
        brightness = 5*cv2.getTrackbarPos('Brightness', 'Parameters') 
        contrast = 5*cv2.getTrackbarPos('Contrast', 'Parameters') 
        self.vid.set(cv2.CAP_PROP_POS_FRAMES,start)
        err,img = self.vid.read()
        effect = controller(img, brightness, contrast) 
      
        
        cv2.imshow('Frame', effect) 
        pass
    
    
    

#----------------------------------------------------------------------#
#                   LANCEMENT DE L'APPLICATION                         #-------------------------------------------------------
#----------------------------------------------------------------------#

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")