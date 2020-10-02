#!usr/bin/python
# -*- coding: utf-8 -*-

from collections import defaultdict

class UniqueDict(dict):
    
    '''
    
    
        a dictionary stucture with specificities:
           -> node is added ones
           -> structure: (id_node, (id_parent, list_values))
                      - id_node: represents node id
                      - id_parent : represents node's id parent
                      - list_values : list of names(images for exemple) that id_node is attributed to
                                      initial is [] empty list
                                      
                                      
    '''  
    
    def __setitem__(self, key, value):
        # value is a tuple (parent, name)
        # with name as the image name for exemple
        
        if key not in self:
            #initialize an array of values
            id, _ = value # get id parent 
            value = (id,[]) # initialize value attribut with an empty list
            dict.__setitem__(self, key, value)
            
        else:
            # in case the key exist(node is already created)
            # i suppose that there is no multi heritage
            # add the new value to the existed values
            _, val = value
            self[key][1].append(val) # [1] to change 'value' attributs 
        
       



class Database(object):

    def __init__(self, root):
        # root corresponds to the name
        # of the first node
            
        self.graph = UniqueDict()
        self.root = root
            
        ##  Graph initialization
        self.add(root, None, None) # None as id, root has no parent
            
        # instantiate an other graph to keep the state of the graph at t <t'
        # i prefered to store it as a graph rather then a list because in same 
        # cases the insertion action may not be executed,
        # insertion of the same node multiple times
        self.old_graph = self.graph.copy()
        
        # save extract
        # my choice was leaned on the backup of extract because in the non-existent 
        # case it will be difficult to recover this data. the other solution was to 
        # browse the two graphs afterwards to recover the difference, this solution 
        # will indeed cost a lot of time and memory space
        self.extract = {}
            
            
            


    def add(self, id_node, id_parent, value=None):               
            self.graph[id_node] = (id_parent, value)
            
            
            
            

    def add_nodes(self, insertions):
        # Take  a list of tuples as input and edit the graph
        
        # store ancient graph
        self.old_graph = self.graph.copy()
        
        # add nodes to the graph
        for id_node, id_parent in insertions:
            self.add(id_node, id_parent) # value at insertion stage is [](to designate a leaf
                                         # which may become an intern node)


                
    def add_extract(self, dict_information ):
        # take a dictionary and store the information
        
        # save added nodes
        self.extract = dict_information
        
        for key, ids_list in dict_information.items(): # key here is value name
            for id in ids_list:
                if id in self.graph.keys():
                    self.graph[id][1].append(key)  ## [1] to get 'value' attribut
                
    
    
      
         
        
    
    def get_extract_status(self):
        
        status_dict = {}
        # get at first added extract values:{ name, [id_node,id_node]}
        for name, list_id in self.extract.items(): # browse added names
            
            # compute added node
            diffrence_dict = { k : self.graph[k] for k in set(self.graph) - set(self.old_graph) }
            
            status = "valid"
            for id_node in list_id:
                ## case of invalid node id
                if id_node not in self.graph.keys():
                    status = "invalid"
                    break
                
                ## current node id already exists in the database 
                else:
                    # check whether id_node has children in the database
                    parent_ids = [list(diffrence_dict.values())[i][0] for i in range(len(diffrence_dict.values()) )]
                    
                    if id_node in parent_ids and status != "coverage_staged":
                        status = "granularity_staged"
                    
                    else:
                        # get parent_id of current node
                        # again i suppose that there is no
                        # multiple heritage
                        parent_id = self.graph[id_node][0] # [0] parent id
                    
                        # check whether parent_id has an other child other than current node
                        if parent_id in parent_ids:
                            status = "coverage_staged"
                    
            status_dict[name] = status
        
        
        return status_dict       
                
