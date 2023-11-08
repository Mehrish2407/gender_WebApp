import os
import cv2
import matplotlib.image as matimg
from app.FRM import faceRecognitionPipeline

from flask import render_template, request

UPLOAD_FOLDER = 'static/upload'

def index():
    return render_template('index.html')




def app():
    return render_template('app.html')


def genderApp():
    if request.method == 'POST':
        f = request.files['image_name']
        filename = f.filename
        # save an image in ipload folder
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path) #saving img in upload folder
        
        # to get predictions
        pred_img, predictions = faceRecognitionPipeline(path)
        pred_filename = 'prediction_img.jpg'
        cv2.imwrite(f'./static/predict/{pred_filename}',pred_img)
        # print(predictions)
        
        #generating a report
        report = []
        for i , obj in enumerate(predictions):
            gray_image = obj['roi'] # grayscale image
            eigen_img = obj['eig_img'].reshape(100,100) #reshaping an eigen image
            gender_name = obj['prediction_name'] #name
            score = round(obj['score']*100,2) #probablity score
            
            # save grayscale eigen image and eigen image in predict folder
            gray_image_name =f'roi_{i}.jpg'
            eig_image_name = f'eigen_{i}.jpg'
            matimg.imsave(f'./static/predict/{gray_image_name}',gray_image, cmap='gray')
            matimg.imsave(f'./static/predict/{eig_image_name}',eigen_img, cmap='gray')
        
            #save report
            report.append([gray_image_name,
                          eig_image_name,
                          gender_name,
                          score])
            
        return render_template('gender.html',fileupload=True, report=report) #POST REQUEST
               
    return render_template('gender.html',fileupload=False) #Get Request


