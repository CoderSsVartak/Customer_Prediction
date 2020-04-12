
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


class model_preprocessing:
    
    def __init__(self, dataset_loc, filename):
        
        self.dataset_loc = dataset_loc
        self.filename = filename
        #Change the current working directory to the base directory        
        os.chdir(self.dataset_loc)


    #Read the dataset file whose name is stored in the variable filename
    #start = index of the 1st column of dataset to be used as the feature
    #end = index of the last column of dataset to be used as the feature
    #op_col = index of the output column
    #Return Numpy array of features and output 
    #columns = list of dataset columns to be dropped
    def read_dataset(self, start, end, op_col, columns):
    
        try:
            dataset = pd.read_csv(self.filename) 
            
            if not columns == []:
                
                dataset = self.drop_dataset_columns(dataset, columns)
                
            features, outputs = dataset.iloc[:, start:end].values, dataset.iloc[:, op_col].values
            return features, outputs
        
        except FileNotFoundError:
            return "File Not Found", ''
        
    

    #If columns have to be dropped from the dataset
    #dataset = DataFrame of the CSV file
    #column_names = list of column names to be dropped
    def drop_dataset_columns(self, dataset, column_names):
        
        try:
            dataset =  dataset.drop(column_names, axis=1)
            return dataset
        except KeyError:
            return "Key Not Found"
    
    #Scale the values of the dataset which is stored in numpy array form
    def scale_dataset(self, features):
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features, scaler
    
    #Pass on the features column and the list of indexes to be encoded with the Categorical encoder
    def label_categorical_data(self, features, indexes):
        
        #Dictionary with key as the index and the value as the encoder object applied on it
        encoder = {}
        
        for index in indexes:
            encoder[index] = LabelEncoder()    
            features[:, index] = encoder[index].fit_transform(features[:, index])
            
            
        return features, encoder
      
        
    #Split the dataset into training and testing based on the test size specified(0 < test_size < 1)
    def get_train_test_data(self, features, output, test_size):

        return(train_test_split(features, output, test_size=0.3))      
        
       

"""
Define the following variables:
dataset_loc -> location of dataset in your system
filename -> Name of the csv datafile
column_names -> Name of the columns to be dropped, Enter empty list if nothing is to be dropped
start -> start index of the column in dataset 
end -> end index of the column in dataset
op_col -> output index of the column in the dataset
indexes -> indexes on which Label encoding is to be done.

#Driver Code  
preprocessor = model_preprocessing(dataset_loc, filename)  

features, output = preprocessor.read_dataset(start, end, op_col, column_names)
features, encoder = preprocessor.label_categorical_data(features, indexes)
features, scaler = preprocessor.scale_dataset(features)
x_train, x_test, y_train, y_test = preprocessor.get_train_test_data(features, output, 0.3)        
"""            
            
            
            
            
            
            
            
            
            
