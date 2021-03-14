# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import cv2
import face_recognition
import os
import pickle
import os

os.chdir("C:/GITHUB/easy_facial_recognition-master/easy_facial_recognition-master/Challenge_SISE_Romain/Challenge_SISE/")
#image = face_recognition.load_image_file("your_file.jpg")
#face_landmarks_list = face_recognition.face_landmarks(image)


# from os import listdir
# from os.path import isfile, join
# onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]



# face_code={}
# for img in onlyfiles :
    
#     picture = cv2.imread(mypath+'/'+img)
#     wb = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
#     boxes = face_recognition.face_locations(wb,model='hog')
#     face_encoding = face_recognition.face_encodings(wb,boxes)[0]#face_recognition.face_encodings(picture)
#     face_code[img.split('.')[0]]=face_encoding

# print(face_code)

# noms=list(face_code.keys())

# image="C:/Users/rpic/Documents/Challenge_SISE/training/BAROU.jpg"
# img_test = cv2.imread(image)
# for key, value in face_code.items():
#     wb = cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB)
#     boxes = face_recognition.face_locations(wb,model='hog')
#     features_test = face_recognition.face_encodings(wb,boxes)[0]#face_recognition.face_encodings(picture)
#     results = face_recognition.compare_faces([value], features_test)
#     if results[0]:
#         print("I see "+key)

video_name="video_input/VIDEO3.mp4"
data = pickle.loads(open('face_enc', "rb").read())
#find path of xml file containing haarcascade file 
cascPathface = os.path.dirname(
 cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
# load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier(cascPathface)
 
video_capture = cv2.VideoCapture("C:/GITHUB/easy_facial_recognition-master/easy_facial_recognition-master/PHOTOS+VIDEOS_SISE-20210310T090923Z-001/PHOTOS+VIDEOS_SISE/"+video_name)
fps = video_capture.get(cv2.CAP_PROP_FPS)
width  = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) )
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
# loop over frames from the video file stream

fourcc = cv2.VideoWriter_fourcc(*'MPEG')
output_movie = cv2.VideoWriter(video_name.split('_')[0]+'_output/'+video_name.split('/')[1].split('.')[0]+'_output.avi', fourcc, fps, (width, height) )

length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

def onChange(trackbarValue=0):
    start = cv2.getTrackbarPos('start',"Parameters")
    video_capture.set(cv2.CAP_PROP_POS_FRAMES,start)
    err,img = video_capture.read()
    cv2.imshow("Frame", img)
    pass

def BrightnessContrast(brightness=1): 
  
    # getTrackbarPos returns the  
    # current position of the specified trackbar.
    start = cv2.getTrackbarPos('start',"Parameters")
    brightness = 5*cv2.getTrackbarPos('Brightness', 'Parameters') 
    contrast = 5*cv2.getTrackbarPos('Contrast', 'Parameters') 
    video_capture.set(cv2.CAP_PROP_POS_FRAMES,start)
    err,img = video_capture.read()
    effect = controller(img, brightness, contrast) 
  
    cv2.imshow('Frame', effect) 
    pass


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



cv2.namedWindow('Frame',cv2.WINDOW_KEEPRATIO&cv2.WINDOW_NORMAL)
cv2.namedWindow('Parameters',cv2.WINDOW_AUTOSIZE&cv2.WINDOW_FREERATIO)
#cv2.resizeWindow('Parameters', 400,250)

cv2.createTrackbar( 'start', "Parameters", 0, length, onChange,)
cv2.createTrackbar( 'end'  , "Parameters", 10, length, onChange)
cv2.createTrackbar('Brightness', 'Parameters', int(255/5), int(255*2/5), BrightnessContrast)
cv2.createTrackbar('Contrast', 'Parameters', int(127/5), int(127*2/5), BrightnessContrast)


onChange(0)
BrightnessContrast(0)
cv2.waitKey()

start = cv2.getTrackbarPos('start',"Parameters")
end   = cv2.getTrackbarPos('end',"Parameters")

brightness = 5* cv2.getTrackbarPos('Brightness', 'Parameters') 
contrast = 5* cv2.getTrackbarPos('Contrast', 'Parameters')

if start >= end:
    raise Exception("start must be less than end")

video_capture.set(cv2.CAP_PROP_POS_FRAMES,start)
cv2.destroyAllWindows()
cv2.namedWindow('Process',cv2.WINDOW_KEEPRATIO&cv2.WINDOW_NORMAL)


k=0
while True:
    k+=1
    
    if video_capture.get(cv2.CAP_PROP_POS_FRAMES) >= end:
        break
    # grab the frame from the threaded video stream
    ret, frame = video_capture.read()
    cv2.imshow("Process", frame)
    adjust_img =  controller(frame, brightness, contrast,settings=False)
    rgb = cv2.cvtColor(adjust_img, cv2.COLOR_BGR2RGB)
    #gray = cv2.cvtColor(adjust_img, cv2.COLOR_BGR2GRAY)
    #faces = faceCascade.detectMultiScale(gray,
    #                                     scaleFactor=1.1,
    #                                     minNeighbors=5,
    #                                     minSize=(50, 50),
    #                                     flags=cv2.CASCADE_SCALE_IMAGE)

    faces = face_recognition.face_locations(rgb,model='hog')
    # the facial embeddings for face in input
    encodings = face_recognition.face_encodings(rgb,faces)
    names = []
    # loop over the facial embeddings incase
    # we have multiple embeddings for multiple fcaes
    for encoding in encodings:
       #Compare encodings with encodings in data["encodings"]
       #Matches contain array with boolean values and True for the embeddings it matches closely
       #and False for rest
        matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=.5)
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
        # loop over the recognized faces
        for ((x, y, w, h), name) in zip(faces, names):
            # rescale the face coordinates
            # draw the predicted face name on the image
            cv2.rectangle(adjust_img, (y, x), (h, w), (0, 255, 0), 2)
            cv2.putText(adjust_img, name, (h, x), cv2.FONT_HERSHEY_SIMPLEX,
             0.75, (0, 255, 0), 2)
    output_movie.write(adjust_img)
    cv2.putText(adjust_img, str(k)+'/'+str(end-start), (0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Process", adjust_img)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
output_movie.release()
cv2.destroyAllWindows()
