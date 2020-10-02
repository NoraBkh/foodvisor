import os
import numpy as np
import csv
import json
from PIL import Image

CSV_FILE = "images/label_mapping.csv"
JSON_FILE = "images/img_annotations.json"

def read_csv(file):
    ## convert a csv file to a dictionary
    ##### args
    # file: csv name file
    # mydict: dictionary (id_object, name_object)
    
    with open(file, mode='r') as infile:
        reader = csv.reader(infile)
        
        ## delete parenthesis to avoid error classification and put everything in lowercase.
        ## the choice was made on the French language because by looking at the data set the
        ## description of the dishes is more detailed in French than in English
        mydict = {rows[0]:(rows[1].split('(')[0]).lower() for rows in reader}
            
    return mydict


def get_label(img_name):
    
    ## returns image's label
        ### 0: no tomato
        ### 1: there is a tomato
    ## for that we need to parse img_annotation and label_mapping
    # get all object ids
    label_dict = read_csv(CSV_FILE)
    
    ### After that, parse json file to get the id of the objects present in the image    
    with open(JSON_FILE) as file:
        data = json.load(file)
        
    
    ## we consider that all the images in the dataset are correctly labeled
    boxes = data[img_name]
    
    for box in boxes:
        id = box['id'] ## object id
        ## check whether the object is a tomato
        if "tomate" in label_dict[id] or "tomates" in label_dict[id]:
            return 1
    
    # Otherwise there are no tomato in the image    
    return 0