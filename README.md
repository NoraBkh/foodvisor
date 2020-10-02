# Dependencies && Install
*Python >= 3.

you need to install all packages in 'requirements.txt' file for the two assignment.

# Assignment1 - Food database

### Description

In this first assignment, the goal was to define a data structure(Database) that handles coverage and granularity of existing labeled data using a directed graph. For that, I created a structure(named: UniqueDict) that creates a graph with unique identifier. So, th structure inherits from the dictionary class of python. this last has as key an identifier of the node, then the value is a tuple (id_parent, list_names) with:
*id_parent: represents the identifier of node's parent. 
*list_name: is a list of object names assigned to this label / identifier.

which leads to this structure: {id_node: (id_parent,list_name) , id_node: (id_parent,list_name) ...}




# Assignment2 - computer vision - Tomato Detection


### Tomato detection using a pre-trained model
In this second assignment, the goal is to detect tomato presence using a deep learning model. For that, i had two model ideas, the first is object detection and the second is to use a deep classifier. As the task is a binary classification so the most judicious solution was to lean towards the second solution.
More details, I used a pre-trained model which is resnet50 i also added two layers at the end  so that the classification corresponds to the binary classification task. Data are images of dishes taken by users. To train the model, just run in a terminal the following command: 
' python main --path path --epochs nb_epoch --batch-size batch_size -- lr learning_rate --cuda '

with:
*path: data/images path
*epochs: number of training epochs
*batch-size; corresponds to the size of batch used for training
*lr: value of learning rate
*if you add --cuda that means  that you will need GPU to train your model else CPU


To visualize loss, accuracy graphes and showing some results, please check data_visualization.jpynb 


