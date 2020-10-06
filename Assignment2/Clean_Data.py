import os
from utils_label import get_label
from PIL import Image

def select_data(path):
    '''
        The function create a balanced dataset by 
        taking half images containing tomatoes and 
        the other half without tomatoes
        
        ## path of original data
    '''
    
    Total = 0
    count = 0
    imgs = list(sorted(os.listdir(os.path.join(path, 'assignment_imgs'))))
    print(len(imgs))
    for idx in range(len(imgs)):
        img_path = os.path.join(path, "assignment_imgs", imgs[idx])
        label = get_label(imgs[idx])
        if label==1:
            img = Image.open(img_path).convert("RGB")
            img.save("imgs/"+imgs[idx])
            print("found a tomato")
            Total+=1
        
        else:
            if count<600:
                img = Image.open(img_path).convert("RGB")
                img.save("imgs/"+imgs[idx])
                count+=1
                
    print("Total= ", Total)
            
select_data('/home/nora/Téléchargements/test_foodvisor/Assignment2')
