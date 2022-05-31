#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 06:00:21 2020

@author: rajeshkanna and thrinad
"""
import json
import os
import shutil
import glob
import os.path
import numpy as np
import time
import tensorflow as tf
import tensorflow_hub as hub
from flask import Flask
from werkzeug.utils import secure_filename
from annoy import AnnoyIndex
from scipy import spatial
import os

def jsontotable(d):
    d = json.loads(d)
    i = 1
    # dirname = os.path.dirname(os.path.realpath(__file__))
    # imgpath = os.path.join(dirname, 'static', 'img')
    # srcpath = os.path.join(dirname, 'chemistry')
    # for img in os.listdir(imgpath):
        # os.unlink(os.path.join(dirname, 'static', 'img', img))
    txt = '''<tr style="text-align: center; font-weight: bold;"><td width="45%">Source Image</td><td width="45%">Similar Images</td><td width="10%">% of similarity</td></tr>'''
    for k, v in d.items():
        # need to copy the k from chid to img path
        if i == 1:
            txt = txt + """<tr style="text-align: center;vertical-align: middle;"><td><p><img src="/static/img/{0}" /><br/>{1}</p></td>""".format(k, k)
            i = i + 1
        elif i == 2:
            txt = txt + """<td><p><img src="/static/img/{0}" /><br/>{1}</p></td><td>{2}%</td></tr>""".format(k, k, str(v))
            i = i + 1
        else:
            txt = txt + """<tr style="text-align: center;vertical-align: middle;"><td></td><td><p><img src="/static/img/{0}" /><br/>{1}</p></td><td>{2}%</td></tr>""".format(k, k, str(v))
    return '<table style="width: 100%;">' + txt + "</table>"

def input_2_existing_data(input_path):
    import json,uuid,os,shutil
    d={}
    for dirs,subdirs,files, in os.walk(input_path):
        for filename in files:
            d["imageName"]= filename.split(".")[0]
            # d["imageName"]= filename
            d["productId"]= str(uuid.uuid4().fields[-1])

    with open("/home/luminad/Documents/Search_Similar_Images/Input/input_json.json",mode="r+") as fp:
        for file in fp:
            seen =json.loads(file)
            seen.append(d)
    with open("/home/luminad/Documents/Search_Similar_Images/Input/input_json.json",mode="w") as fp:
        json.dump(seen,fp)
   
    return(d)

def read_feature_vectors(path):
    import os
    All_fiiles_chem=[]
    for dirs,subdirs,files, in os.walk(path):
        for filename in files:
            All_fiiles_chem.append(os.path.join(dirs,filename).split(".")[0]+".npz")
    return(All_fiiles_chem)

def remove_simlarity_Image_duplicate():
    import json
    with open('/home/luminad/Documents/Search_Similar_Images/output/chemistry_nearest_neighbors.json', 'r') as fp:
        for file in fp:
            seens = json.loads(file)
            for i in range(len(seens)):
                if seens[i]["master_pi"] == seens[i]["similar_pi"]  :
                    seens.pop(i)
                    break
    with open('/home/luminad/Documents/Search_Similar_Images/output/chemistry_nearest_neighbors.json', 'w') as fp:
        json.dump(seens, fp)
        

def finding_extensions(find,inames):
    ext_list = []
    for i in find:
        if i+".jpg" in inames:
            ext_list.append(i+".jpg")
        elif i+".JPG" in inames:
            ext_list.append(i+".JPG")
        elif i+".JPEG" in inames:
            ext_list.append(i+".JPEG")
        elif i+".jpeg" in inames:
            ext_list.append(i+".jpeg")
        elif i+".PNG" in inames:
            ext_list.append(i+".PNG")
        elif i+".png" in inames:
            ext_list.append(i+".png")

    return (ext_list)

def input_data_preparation(path,output_path):
    import json,uuid,os
    lists = []
    for dirs,subdirs,files, in os.walk(path):
        for filename in files:
            lists.append({"imageName":filename.split(".")[0],"productId":str(uuid.uuid4().fields[-1])})
            # lists.append({"imageName":filename,"productId":str(uuid.uuid4().fields[-1])})
    with open(output_path+"/input_json.json",mode="w",encoding='utf-8') as fp:
        json.dump(lists,fp)

def load_img(path):
  img = tf.io.read_file(path)
  img = tf.io.decode_jpeg(img, channels=3)
  img = tf.image.resize_with_pad(img, 224, 224)
  img  = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
  return img

def get_image_feature_vectors(input_path):
    import shutil
    i = 0
    start_time = time.time()
    print("---------------------------------")
    print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Started at %s" %time.ctime())
    print("---------------------------------")
    # Definition of module with using tfhub.dev handle    
    # module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"
    module_handle = "https://tfhub.dev/google/imagenet/mobilenet_v2_140_224/feature_vector/4"

    # Load the module
    module = hub.load(module_handle)
    print("---------------------------------")
    print ("Step.1 of 2 - mobilenet_v2_140_224 - Loading Completed at %s" %time.ctime())
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
    print("---------------------------------")
    print ("Step.2 of 2 - Generating Feature Vectors -  Started at %s" %time.ctime())
    # Loops through all images in a local folder
   
    # for dirs,subdirs,files, in os.walk("D:/FOC/16-09-2020/15-09-2020/Hexagon/chemistry/"):
    for dirs,subdirs,files, in os.walk(input_path):
        for filename in files:
            print(filename )
            if filename.endswith(".jpeg") or filename.endswith(".JPEG"):
                print(filename)
                i = i + 1
                img = load_img(os.path.join(dirs,filename))
            elif filename.endswith(".png") or filename.endswith(".PNG"):
                print(filename)
                i = i + 1
                img = load_img(os.path.join(dirs,filename))
            elif filename.endswith(".jpg") or filename.endswith(".JPG"):
                print(filename)
                img = load_img(os.path.join(dirs,filename))
                i = i + 1
            print("-----------------------------------------------------------------------------------------")
            print("Image count                     :%s" %i)
            print("Image in process is             :%s" %filename)
            # Loads and pre-process the image
            # img = load_img(filename)
            # Calculate the image feature vector of the img
            features = module(img)  
            # Remove single-dimensional entries from the 'features' array
            feature_set = np.squeeze(features)  
            # Saves the image feature vectors into a file for later use
            # outfile_name = os.path.basename(filename).split('.')[0] + ".npz"
            # print(filename.split('.')[-2])
            outfile_name = filename.split('.')[0] + ".npz"
           
            # out_path = os.path.join("D:/refresh/feature-vectors/chemisty/", outfile_name)
            out_path = os.path.join("/home/luminad/Documents/Search_Similar_Images/feature vectors/", outfile_name)
            # Saves the 'feature_set' to a text file
            np.savetxt(out_path, feature_set, delimiter=',')
            shutil.move(os.path.join(dirs,filename),"/home/luminad/Documents/Search_Similar_Images/static/img/")
            print("Image feature vector saved to   :%s" %out_path)
            print("---------------------------------")
            print ("Step.2 of 2 - Generating Feature Vectors - Completed at %s" %time.ctime())
            print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))
            print("--- %s images processed ---------" %i)

def match_id(filename):
    # store_the_input = "C:/Users/trinadh.n/Arty_Image_Detection_Engine/input"
    # "D:/Arty/Arty_data/Production_data_Hexagon.json"
    with open("/home/luminad/Documents/Search_Similar_Images/Input/input_json.json") as json_file:  
  # with open('D:/Arty/Arty_data/Production_data_Hexagon.json') as json_file:
   
        for file in json_file:
            seen = json.loads(file)
            print(seen[0:21])
   
            for line in seen:
             
              if filename==line['imageName']:
                print(line)
                # print(line['productId'])
                return line['productId']
                break

def cluster():
 
  start_time = time.time()
 
  print("---------------------------------")
  print ("Step.1 - ANNOY index generation - Started at %s" %time.ctime())
  print("---------------------------------")

  # Defining data structures as empty dict
  file_index_to_file_name = {}
  file_index_to_file_vector = {}
  file_index_to_product_id = {}

  # Configuring annoy parameters
  dims = 1792
  n_nearest_neighbors = 20
  trees = 10000

  # Reads all file names which stores feature vectors
  # allfiles = glob.glob('D:/refresh/feature-vectors/chemisty/*.npz')
  allfiles  = read_feature_vectors("/home/luminad/Documents/Search_Similar_Images/feature vectors/")
  # allfiles  = read_feature_vectors(Input_Images_path)
 
  len(allfiles)
  t = AnnoyIndex(dims, metric='angular')

  for file_index, i in enumerate(allfiles):
   
    # Reads feature vectors and assigns them into the file_vector
    file_vector = np.loadtxt(i)

    # Assigns file_name, feature_vectors and corresponding product_id
    file_name = os.path.basename(i).split('.')[0]
    file_index_to_file_name[file_index] = file_name
    file_index_to_file_vector[file_index] = file_vector
    file_index_to_product_id[file_index] = match_id(file_name)

    # Adds image feature vectors into annoy index  
    t.add_item(file_index, file_vector)

    print("---------------------------------")
    print("Annoy index     : %s" %file_index)
    print("Image file name : %s" %file_name)
    print("Product id      : %s" %file_index_to_product_id[file_index])
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))


  # Builds annoy index
  t.build(trees)

  print ("Step.1 - ANNOY index generation - Finished")
  print ("Step.2 - Similarity score calculation - Started ")
 
  named_nearest_neighbors = []

  # Loops through all indexed items
  for i in file_index_to_file_name.keys():

    # Assigns master file_name, image feature vectors and product id values
    master_file_name = file_index_to_file_name[i]
    master_vector = file_index_to_file_vector[i]
    master_product_id = file_index_to_product_id[i]

    # Calculates the nearest neighbors of the master item
    nearest_neighbors = t.get_nns_by_item(i, n_nearest_neighbors)

    # Loops through the nearest neighbors of the master item
    for j in nearest_neighbors:

      print(j)

      # Assigns file_name, image feature vectors and product id values of the similar item
      neighbor_file_name = file_index_to_file_name[j]
      neighbor_file_vector = file_index_to_file_vector[j]
      neighbor_product_id = file_index_to_product_id[j]

      # Calculates the similarity score of the similar item
      similarity = 1 - spatial.distance.cosine(master_vector, neighbor_file_vector)
      rounded_similarity = int((similarity * 10000)) / 10000.0

      # Appends master product id with the similarity score
      # and the product id of the similar items
      named_nearest_neighbors.append({
        'similarity': rounded_similarity,
        'master_pi': master_product_id,
        'similar_pi': neighbor_product_id})

    print("---------------------------------")
    print("Similarity index       : %s" %i)
    print("Master Image file name : %s" %file_index_to_file_name[i])
    print("Nearest Neighbors.     : %s" %nearest_neighbors)
    print("--- %.2f minutes passed ---------" % ((time.time() - start_time)/60))

 
  print ("Step.2 - Similarity score calculation - Finished ")

  # Writes the 'named_nearest_neighbors' to a json file
  with open('/home/luminad/Documents/Search_Similar_Images/output/chemistry_nearest_neighbors.json', 'w') as out:
  # with open('D:/Arty/Arty_data/chemistry_nearest_neighbors.json', 'w') as out:
    json.dump(named_nearest_neighbors, out)

  print ("Step.3 - Data stored in 'nearest_neighbors.json' file ")
  print("--- Prosess completed in %.2f minutes ---------" % ((time.time() - start_time)/60))

import os
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = '/home/luminad/Documents/Search_Similar_Images/upload'
ALLOWED_EXTENSIONS = {'JPEG', 'PNG', 'png', 'jpg', 'jpeg', 'JPG'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('uploaded_file', filename=filename))
    return '''
           <!doctype html>

        <head>

     <title>Upload new File</title>

 

      <link rel="stylesheet"

href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css"

integrity="sha512-dTfge/zgoMYpP7QbHy4gWMEGsbsdZeCXz7irItjcC3sPUFtf0kuFbDz/ixG7ArTxmDjLXDmezHubeNikyKGVyQ=="

crossorigin="anonymous">

     <script

src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js"

integrity="sha512-K1qjQ+NcF2TYO/eI3M6v8EiNYZfA95pQumfvcVrTHtwQVDG+aHRqLi/ETn2uB+1JqwYqVG3LIvdm9lj6imS/pQ=="

crossorigin="anonymous"></script>

     <script type="text/javascript"

src="https://ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>

     <script type="text/javascript"

src="https://www.jquery-az.com/boots/js/bootstrap-filestyle.min.js">

</script>

 

     <meta name="viewport" content="width=device-width, initial-scale=1.0">

     <style>

        .topdiv {

 

        background-color: #7f9cb7;

     color: #FEFDF9;

     font-size: 30px;

     margin: 0px;

     padding-top: 2px;

     border: 1px solid;

     height: 59px;

     font-family: sans-serif, Verdana;

     border-radius: 4px;

}

     </style>

                <script>

                                                function validateForm() {

                                               

                                               

  var x = document.forms["myForm"]["file"].value;

  if (x == "") {

    alert("Please upload a file");

    return false;

  }

}

</script>

     </head>

     <body style="    background-color: white;">

     <div class="topdiv">

<img

src="http://dp2.luminadatamatics.com/iStyleMapping/Images/Datamatics_Logo_2.png"

alt="logo" style="width: 145px;margin-top: 6px;"> </div>

 

<form name="myForm" action="" enctype=multipart/form-data onsubmit="return validateForm()" method="post" onsubmit="return validateForm()">

 

     <div class="container">

 

<div class="col-xs-2">

</div>

     <div class="col-xs-8" style="    margin-top: 13%;">

     <h2 style="align-text:center">Search Similar Images</h2>

         <div class="form-group">

 

             <input type=file id="icondemo" name=file>

 

         </div>

     </div>

<div class="col-xs-2" style="margin-top: 155px;">

<h2 style="align-text:center;display: none;">Lega</h2>

<input type=submit value=Search class="btn btn-success" style="float:left">

</div>

</div>

</form>

<script>

             $('#icondemo').filestyle({

                 iconName : 'glyphicon glyphicon-file',

                 buttonText : 'Select File',

                 buttonName : 'btn-warning'

             });

 

</script>

     <body>

 

</html>
    '''

from flask import send_from_directory
from flask import jsonify
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    path_name = app.config['UPLOAD_FOLDER']
    # input_2_existing_data
    # os.path.basename(filename)
    # input_data = input_2_existing_data( path_name)
       
    # input_path      = "D:/Arty/Arty_data/single image/test2/"
    store_the_input = "/home/luminad/Documents/Search_Similar_Images/Input"
    input_data      = input_2_existing_data(path_name)
    # npz_path        = "C:/Users/trinadh.n/Arty_Image_Detection_Engine/feature vectors"
    # desination_path =  'C:/Users/trinadh.n/Arty_Image_Detection_Engine/chemistry'
    get_image_feature_vectors(path_name)
    # get_image_feature_vectors(path_name)
    def match_id(filename):
    # store_the_input = "C:/Users/trinadh.n/Arty_Image_Detection_Engine/input"
    # "D:/Arty/Arty_data/Production_data_Hexagon.json"
        with open("/home/luminad/Documents/Search_Similar_Images/Input/input_json.json") as json_file:  
      # with open('D:/Arty/Arty_data/Production_data_Hexagon.json') as json_file:
       
            for file in json_file:
                seen = json.loads(file)
                print(seen[0:21])
       
                for line in seen:
                 
                  if filename==line['imageName']:
                    print(line)
                    # print(line['productId'])
                    return line['productId']
                    break

    cluster()
    # c = cluster()
    # store_the_input = "D:/output"
    import json
    # def remove_simlarity_Image_duplicate():
    #     import json
    #     with open('C:/Users/trinadh.n/Documents/testing.json', 'r') as fp:
    #         for file in fp:
    #             seen = json.loads(file)
    #             for i in range(len(seen)):
    #                 if seen[i]["master_pi"] == seen[i]["similar_pi"]:
    #                     seen.pop(i)
    #                     break
           
    #     with open('C:/Users/trinadh.n/Documents/testing.json', 'w') as fp:
    #         json.dump(seen, fp)
    # import json
    # with open('C:/Users/trinadh.n/Arty_Image_Detection_Engine/output/chemistry_nearest_neighbors.json', 'r') as fp:
    #     for file in fp:
    #         seen = json.loads(file)
    #         for i in range(len(seen)):
    #             if seen[i]["master_pi"] == seen[i]["similar_pi"]:
    #                 seen.pop(i)
    #                 break
   
    # with open('C:/Users/trinadh.n/Arty_Image_Detection_Engine/output/chemistry_nearest_neighbors.json', 'w') as fp:
    #     json.dump(seen, fp)
    # remove_simlarity_Image_duplicate()
    similar_id = []
    percentage = []
    
    import json
    with open('/home/luminad/Documents/Search_Similar_Images/output/chemistry_nearest_neighbors.json', 'r') as fp:
        for file in fp:
            seen = json.loads(file)
            for line in seen:
                if line["master_pi"]==input_data["productId"]:
                    similar_id.append(line['similar_pi'])
                    percentage.append(line['similarity'])
    
        image_list = [i for i in os.listdir("/home/luminad/Documents/Search_Similar_Images/static/img")]
        imagename=[]
        for i in similar_id:
            with open("/home/luminad/Documents/Search_Similar_Images/Input/input_json.json",mode='r') as fp:
               
                for file in fp:
                    seen = json.loads(file)
                    for line in seen:
                        if line["productId"]==i:
                            imagename.append(line["imageName"])
                            # print(line["imageName"])
        percentage = [round(i*100,2) for i in percentage]
        # percentage = [str(i)+"%" for i in percentage]
        image_names_ext = finding_extensions(imagename,image_list)
        d = dict(zip(image_names_ext,percentage))
        d=json.dumps(d  , sort_keys=False)
    return '''<!doctype html>
       <head>
    <title>Upload new File</title>

     <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/css/bootstrap.min.css" integrity="sha512-dTfge/zgoMYpP7QbHy4gWMEGsbsdZeCXz7irItjcC3sPUFtf0kuFbDz/ixG7ArTxmDjLXDmezHubeNikyKGVyQ==" crossorigin="anonymous">
    <script src="https://maxcdn.bootstrapcdn.com/bootstrap/3.3.5/js/bootstrap.min.js" integrity="sha512-K1qjQ+NcF2TYO/eI3M6v8EiNYZfA95pQumfvcVrTHtwQVDG+aHRqLi/ETn2uB+1JqwYqVG3LIvdm9lj6imS/pQ==" crossorigin="anonymous"></script>
    <script type="text/javascript" src="//ajax.googleapis.com/ajax/libs/jquery/1.9.1/jquery.min.js"></script>
    <script type="text/javascript" src="https://www.jquery-az.com/boots/js/bootstrap-filestyle.min.js"> </script>

    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
       .topdiv {

       background-color: #7f9cb7;
    color: #FEFDF9;
    font-size: 30px;
    margin: 0px;
    padding-top: 2px;
    border: 1px solid;
    height: 59px;
    font-family: sans-serif, Verdana;
    border-radius: 4px;
    width:108%;
}
.table {
   margin: auto;
   width: 65% !important;
}
    </style>
    </head>
    <body style="    background-color: white;">
    <div class="topdiv">
<img src="http://dp2.luminadatamatics.com/iStyleMapping/Images/Datamatics_Logo_2.png" alt="logo" style="width: 145px;margin-top: 6px;">
</div>
    <center><h2 style="align-text:center">Reporting Similar Images</h2></center><br/>
<div id="pricing" class="container-fluid">''' + jsontotable(d) + '''</div>

    </body></html>'''
    # return send_from_directory(os.path.join(app.config['UPLOAD_FOLDER']))
   
# if __name__ == '__main__':
#     app.debug = True
#     app.run(host='172.16.1.194', port=5000)        

if __name__ == '__main__':
    app.debug = True
    app.run(host = '0.0.0.0',port=5005)