# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 09:29:22 2020

@author: Tsega
"""
"""
The configuration parameters are described as follows:
    encoder_node_size-> The dimension of the latent space
    Input_activ -> Input layer activation
    Hidden_activ-> Hidden layers activation 
    filter_size -> Convolutional layers filter kernel size
    polling_size-> polling size for max polling layers
    batch_size-> Batch factor for training, i.e., Batch=input data row dimension/batch_size
    File_loc-> location of file for training, i.e., the UCR data set locatin
    File_name-> Name of file for training, i.e., excluding the _TRAIN.tst or _TEST.tsv substring 
    classif_mode-> 0= All classes within the data set are used for training, 
                   1= Training is conducted on a selected class
    select_all -> 0=only data sets within the xxx_TRAIN.tsv are used for training the network
                  1=data sets witin xxx_TEST.tsv and xxx_TRAIN.tsv are mearged together to train the network
    Class_lable-> index of the class selected for training, if per class training is issued
    First_time_training -> 1= Model is built and traind from scratch
                           0= Model is built but weights are loaded with a xxx.ckpt file located in model save path directory
    Model_Save_Path -> Directory for xxx.ckpt file which could be used to load model with weights 
    Model_save_name -> the name of the xxx.ckpt file, i.e., xxx=Model_save_namexxxx.ckpt
    
"""
class Warping_Reg_Net_Config:
    encoder_node_size=0
    Input_activ=''
    Hidden_activ=''
    encoder_node_size=0
    filter_size=3
    polling_size=3
    En_L1_reg=0
    En_L2_reg=0
    De_L1_reg=0
    De_L2_reg=0
    Cl_L1_reg=0
    Cl_L2_reg=0
    start_at=''
    batch_size=1
    Epoch=600
    Learning_rate=0.00001
    File_loc=''
    File_name=''
    Model_save_path=''
    Model_save_name=''
    classif_mode=1
    select_all=1
    class_label=0
    First_time_train=0
    model_type=0
    validation_size=0
    components=0
    run_per_single_data=25
    Quantiles=[]
    To_start_from=''
    To_start_from_FName=''
    def __init__(self):
            self.encoder_node_size=0
            self.Input_activ='linear'
            self.Hidden_activ='relu'
            self.encoder_node_size=0
            self.filter_size=3
            self.polling_size=3
            self.start_at='Start_at.csv'
            self.run_per_single_data=25
            self.En_L1_reg=0.0000
            self.En_L2_reg=0.0000
            self.De_L1_reg=0.0000
            self.De_L2_reg=0.0000
            self.Cl_L1_reg=0.0000
            self.Cl_L2_reg=0.0000
            self.batch_size=4
            self.Epoch=1500
            self.Quantiles=[[0.15,0.85],[0.25,0.75],[0.35,0.65],[0.5,0.5]] # [[0.85,0.85],[0.75,0.75],[0.65,0.65],[0.5,0.5],[0.15,0.15],[0.25,0.25],[0.35,0.35]]
            self.Learning_rate=1e-4
            self.File_loc='' #The Directory where the UCR datasets are stored 
            self.File_name=''
            self.Model_save_path='' # The directory where you want to save the weights of the trained model
            self.List_of_data_sets='' # The directory where "Dataset_List.csv" is stored 
            self.List_of_data_sets_FName='Dataset_List.csv'
            self.classif_mode=0
            self.To_start_from='' #The directory where "Startindex.csv" is stored
            self.To_start_from_FName='Startindex.csv'
            self.select_all=0
            self.class_label=1
            self.First_time_train=1
            self.validation_size=0.20
            self.model_type=2
