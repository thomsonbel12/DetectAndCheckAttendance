
import cv2,io
import os

import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

from flask import Flask,render_template,request,Response
app=Flask(__name__)
route=os.path.dirname(os.path.abspath(__file__))
des=os.path.join(route,"TrainingImage")
sampleimage=0
@app.route("/",methods=["GET"])
def main():
    return render_template ("index.html")
@app.route("/<Id>&<name>",methods=["POST","GET"])
@app.route("/layidname",methods=["POST","GET"])
def layidname():
    sampleimage=0
    Id_put=request.form["id_input"]
    name_put=request.form["name_input"]
    return render_template("cam.html",Ids=Id_put,names=name_put)
    
@app.route("/cam")
def cam():

    return render_template("cam.html")
@app.route('/process')
def process():
	return render_template('process.html')

def TakeImages(Ids,names):        
    Id=Ids
    name=names
    #if(is_number(Id) and name.isalpha()):
    # cam = cv2.VideoCapture(0)
    vc = cv2.VideoCapture(0)
    harcascadePath = "static\haarcascade_frontalface_default.xml"
    # harcascadePath = "C:\Program Files\Python37\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    sampleNum=0
    while(True):
        ret, frame = vc.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)    


            sampleNum=sampleNum+1

            cv2.imwrite("NhanDien&DiemDanh\TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #luu anh train vao folder

        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')   

        if sampleNum>100: #luu anh cho den khi dc 100 anh
            break
  
    # res = "Ảnh đã được lưu với ID : " + Id +" - Tên : "+ name
    row = [Id , name]
    with open('NhanDien&DiemDanh\StudentDetails\StudentDetails.csv','a+') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    csvFile.close()
       # message.configure(text= res)
    #else:
        # if (is_number(Id)):
        #     res = "Enter Alphabetical Name"
        #     message.configure(text=res)
        # if (name.isalpha()):
        #     res = "Enter Numeric Id"
        #     message.configure(text=res)
    TrainImages()
    return "ok"
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    harcascadePath = "static\haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = getImagesAndLabels("NhanDien&DiemDanh\TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("NhanDien&DiemDanh\TrainingImageLabel\hile.yml") # lưu model mới train vào thư mục
    
   

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids



def gen():
    vc = cv2.VideoCapture(0)
    """Video streaming generator function."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer() 
    recognizer.read("NhanDien&DiemDanh\TrainingImageLabel\hile.yml")
    harcascadePath = "static\haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("NhanDien&DiemDanh\StudentDetails\StudentDetails.csv")
    #cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names) 
    dem=0
    while True:
        read_return_code, frame = vc.read()
        dem=dem+1
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 50):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+str(aa)
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]      
            else:
                Id='Unknown'                
                tt=str(Id)  
            if(conf > 100):
                noOfFile=len(os.listdir("NhanDien&DiemDanh\ImagesUnknown"))+1
                cv2.imwrite("NhanDien&DiemDanh\ImagesUnknown\Image"+str(noOfFile) + ".jpg", frame[y:y+h,x:x+w])            
            cv2.putText(frame,str(tt),(x,y+h), font, 1,(255,255,255),2)   
        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        
        if (dem==50):
            ts = time.time()      
            date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
            timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
            Hour,Minute,Second=timeStamp.split(":")
            fileName="NhanDien&DiemDanh\Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
            attendance.to_csv(fileName,index=False)
            
    
    # cam.release()
    # cv2.destroyAllWindows()
    #print(attendance)
    
        
        
    
        #cam.release()
        #cv2.destroyAllWindows()
        

@app.route("/output",methods=["GET"])
def output():
    return render_template("output.html")
@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        gen(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
def video():
    vc = cv2.VideoCapture(0)
    while True:
        
        read_return_code, frame = vc.read()
        encode_return_code, image_buffer = cv2.imencode('.jpg', frame)
        io_buf = io.BytesIO(image_buffer)
        yield(b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')   
        

@app.route('/video_feed1',methods=["POST","GET"])
def video_feed1():
    """Video streaming route. Put this in the src attribute of an img tag."""
    if request.method=="POST": 
        return Response(
            TakeImages(Ids=request.values["id_input"],names=request.values["name_input"]),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )
    return Response(
           video() ,
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )



if __name__=='__main__':
    app.run(host="0.0.0.0",port=5000)