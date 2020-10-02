from torchvision import models
import torch.nn as nn
import torch


device = torch.device("cuda" if torch.cuda.is_available() 
                                  else "cpu")
def get_model(num_classes):
    
    '''
        get an instance of the model
        
        # for that we use a pretrained model with is resnet50
        # as we only have two classes, we had to modify the last fully connected
        
        ## params
            # num_classes: represent number of objets/classes
    '''
    
    # Pre-trained model
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, num_classes)
                             ## softmax is included in the loss,
                             ## this is the reason why I did not 
                             ## add a softmax layer
                            )
    
    return model.to(device)
    