# Pytorch-tutorial
This is a coding repo for pytorch tutorial. It works on CPU.
This repo mainly focus on linear model, softmax model, cnn model and inception model

# Requirement
  - Pytorch
  - python2 or python3
  - numpy
  
# Usage
## linear_model.py
  
  (1) build dataset
  
  - This model requires a .csv datasets with format like(1,2,3,4,5,6,7,8,0), each row is seperated by "," and each row contains a (data,label), the first 8 dims are input data while the last dim is considered as a label for this data.
    
  - after you build such a ".csv" file then store it in data directory(named data) it is time to begin to run the model
    
  (2) training and testing
  
      $ python linear_model.py

## softmax_classifier.py
  
  (1) build a directory named data and then build the sub-directory in data named mnist
  
      $ mkdir data
      $ cd data
      $ mkdir mnist
      
  (2) training & testing
  
      $ python softmax_classifier.py

## simple_cnn.py
      
  (1) training & testing
  
      $ python simple_cnn.py

## inception_v1.py
      
  (1) training & testing
  
      $ python incpetion_v1.py

# Contacts
  - Email:computerscienceyyz@163.com
