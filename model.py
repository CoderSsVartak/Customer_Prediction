
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


#Enter the dataset filename and the location of the file 
dataset_loc = 'C:/Users/LENOVO/Desktop/coding/Projects/BankingCustomersPredictor-master'
filename = 'Churn_Modelling.csv'


class model_preprocessing:
    
    def __init__(self, dataset_loc, filename):
        
        self.dataset_loc = dataset_loc
        self.filename = filename
        #Change the current working directory to the base directory        
        os.chdir(dataset_loc)


    #Read the dataset file whose name is stored in the variable filename
    #start = index of the 1st column of dataset to be used as the feature
    #end = index of the last column of dataset to be used as the feature
    #op_col = index of the output column
    #Return Numpy array of features and output 
    def read_dataset(filename, start, end, op_col):
    
        try:
            dataset = pd.read_csv(filename) 
            features, outputs = dataset.iloc[:, start:end].values, dataset.iloc[:, op_col].values
            return features, output
        
        except FileNotFoundError:
            return "File Not Found"
        
    
column_names = ['Geography']
start, end, op_col = 2, 12, -1
indexes = [1, 2]

    #If columns have to be dropped from the dataset
    #dataset = DataFrame of the CSV file
    #column_names = list of column names to be dropped
    def drop_dataset_columns(dataset, column_names):
        
        try:
            dataset =  dataset.drop(column_names, axis=1)
            return dataset
        except KeyError:
            return "Key Not Found"
    
    #Scale the values of the dataset which is stored in numpy array form
    def scale_dataset(features):
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        
        return features
    
    #Pass on the features column and the list of indexes to be encoded with the Categorical encoder
    def label_categorical_data(features, indexes):
        
        #Dictionary with key as the index and the value as the encoder object applied on it
        encoder = {}
        
        for index in indexes:
            encoder[index] = LabelEncoder()    
            features[:, index] = encoder[index].fit_transform(features[:, index])
        
        return features, encoder
            

#x_train, x_test, y_train, y_test = train_test_split(features, outputs, test_size=0.3)         
            
            
            
            
            
            
            
            
            
            
            
            