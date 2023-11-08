import numpy as np
import sklearn
import pickle
import cv2

# load all models
haar = cv2.CascadeClassifier('./Model/opencv_haarcascade_frontalface_default.xml') #cascade Classifier
model_svm = pickle.load(open('./Model/model_svm.pickle',mode='rb')) # machine learning model(SVM)
pca_models = pickle.load(open('./Model/pca_dict.pickle',mode = 'rb')) #pca dictionary
model_pca = pca_models['pca'] #pca models
mean_face_arr = pca_models['mean_face'] #mean faces


def faceRecognitionPipeline(filename,path=True):
    if path:
        # S1: read Images
        img = cv2.imread(filename) #read in BGR
    else:
        img = filename #array
    
    #S2: Convert into gray Scale:
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    gray
    #creating an empty list: 
    predictions = []
    #S3: croping an image
    faces = haar.detectMultiScale(gray,1.5,3)
    for x,y,w,h in faces:
        roi = gray[y:y+h,x:x+w]
        #S4: normalize between 0 to 1
        roi = roi/255.0   #0 is black and 255 is white color
        #S5: Resize the images(100*100)
        if roi.shape[1]>100:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_AREA)
        else:
            roi_resize = cv2.resize(roi,(100,100),cv2.INTER_CUBIC)

        # S6: Flattening(1*10000)
        roi_reshape = roi_resize.reshape(1,10000)

        #S7: Subtracting with mean
        roi_mean = roi_reshape - mean_face_arr #subtract face with mean face
        #S8:get an eigen image (apply mean to pca)
        eigen_image = model_pca.transform(roi_mean)
        #S9: Eigen Image for Visualization
        eig_img = model_pca.inverse_transform(eigen_image)
        #S10: pass to ml model (svm) and get prediction
        results = model_svm.predict(eigen_image)
        prob_score = model_svm.predict_proba(eigen_image)
        prob_score_max = prob_score.max()
        #S11: Genrating Report
        text = "%s: %d"%(results[0],prob_score_max*100) 

        # defining the color based on result male = blue || female= pink
        if results[0] == 'male':
            color = (255,255,0)
        else:
            color =(255,0,255)
        cv2.rectangle(img,(x,y),(x+w,y+h),color,2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color,-1) #subtracting with 30pixel
        cv2.putText(img,text,(x,y),cv2.FONT_HERSHEY_PLAIN,4,(255,255,255),7)
        output = {
        'roi':roi,
        'eig_img':eig_img,
        'prediction_name':results[0],
        'score':prob_score_max
        }
        predictions.append(output)
        
    return img, predictions