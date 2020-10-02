from torchvision import models
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable

from PIL import Image

MODEL_NAME = 'Tomatomodel.pth'




def has_tomatoes(img_path):
    
    ## Test first the existance of the file
    try:
        with open(img_path, 'r') as fp: 
            test_transforms = transforms.Compose([transforms.Resize(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                      
                                              ])
            # if we do GPU, switch to CUDA
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model=torch.load(MODEL_NAME)
            
            # put model in evaluation mode
            model.eval()
            
            # read image
            image = Image.open(img_path).convert("RGB")
            image_tensor = test_transforms(image).float() # pre-process the image
            image_tensor = image_tensor.unsqueeze_(0)
            input = Variable(image_tensor)
            input = input.to(device)
            output = model(input)
            index = output.data.cpu().numpy().argmax()
            
            ## Tomato presence is coded with 1
            if index == 1:
                return True
            else:
                return False
            
            
    except IOError:
        print ("Error! file cannot be openned")

    