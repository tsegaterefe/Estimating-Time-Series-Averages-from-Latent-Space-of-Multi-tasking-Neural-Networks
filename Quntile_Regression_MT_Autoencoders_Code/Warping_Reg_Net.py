# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 11:28:01 2021

@author: Tsega
"""
import numpy as npy
import Custom_trainer as trainer
import csv 
from tensorflow.keras.layers import Input
import time
from keras.utils.np_utils import to_categorical
import pickle
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras import optimizers
from tslearn import metrics
class Warping_Reg_Net:
    enc_diff=0
    data=npy.array
    Mix_Dens=''
    data_average=npy.array
    File_loc=''
    File_name=''
    Model_save_path=''
    Model_save_name=''
    classif_mode=0
    uniques=[]
    counts=[]
    estimatedmeans=[]
    confusion_matrix=npy.array
    Encoder_layer=Model
    Mixiture_Layer=Model
    RechakeLaye=Model
    Decoder_layer=Model
    my_mix_network=Model
    Classifier_layer=Model
    Warper_layer=Model
    Right_encoder=Model
    Left_encoder=Model
    mymodel=Model
    training_time=0
    final_estimate=npy.array
    hist_dict={}
    my_loss_count=0
    pridicted=0
    temp=''
    temps=''
    rejected=[]
    time_loss=0
    encoder_node_size=0
    validation=0
    validation_data=npy.array
    my_uniqes_copy=npy.array
    my_normalized=npy.array
    my_normalized_valid=npy.array
    fake_losses=[]
    real_losses=[]
    generator_loss=[]
    facke_acc=[]
    real_acc=[]
    generator_acc=[]
    comp_means=npy.array
    comp_var=npy.array
    comp_pis=npy.array
    comps=0
    kl=0
    my_enc_model=Model
    def __init__(self,File_loc,File_name,Model_save_path, Model_save_name,classif_mode,select_all,class_label,validation):
        #tf.compat.v1.disable_eager_execution()
        self.File_loc=File_loc
        self.File_name=File_name
        self.Model_save_path=Model_save_path
        self.Model_save_name=Model_save_name
        self.classif_mode=classif_mode
        self.my_enc_model=Model
        self.comps=1
        self.comp_means=npy.array
        self.comp_var=npy.array
        self.comp_pis=npy.array
        self.Mix_Dens=''
        self.my_mix_network=Model
        self.estimatedmeans=0
        self.Model_save_path=Model_save_path
        self.Model_save_name=Model_save_name
        self.temp=pd.read_csv(File_loc+File_name+'_TRAIN.tsv',sep='\t',header=None)
        self.temps=pd.read_csv(File_loc+File_name+'_TEST.tsv',sep='\t',header=None)
        self.data=npy.zeros((1,self.temp.shape[1]),dtype=npy.float32)
        data_copy=npy.array(self.temp.iloc[:,:])
        uniques=npy.unique(data_copy[:,0])
        self.fake_losses=[]
        self.real_losses=[]
        self.generator_loss=[]
        self.facke_acc=[]
        self.real_acc=[]
        self.generator_acc=[]
        self.kl=0
        self.enc_diff=0
        count=0
        max_length=max(self.temps.shape[0],self.temp.shape[0])
        if classif_mode==0 and select_all==1:
            for k in range(len(uniques)):
                for i in range(max_length):
                    if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[k]:
                        if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                            self.rejected.append(i)
                        else:
                            count=count+1
                            temps_c=npy.array([self.temp.iloc[i,:]])
                            temps_c=temps_c.reshape((1,temps_c.shape[1]))
                            self.data=npy.concatenate((self.data,temps_c),axis=0)
                    if i<self.temps.shape[0] and self.temps.iloc[i,0]==uniques[k]:
                        if npy.isnan(npy.sum(self.temps.iloc[i,1:])):
                            self.rejected.append(i)
                        else:
                            count=count+1
                            temps_c=npy.array([self.temps.iloc[i,:]]).reshape(1,self.temps.shape[1])
                            self.data=npy.concatenate((self.data,temps_c),axis=0)
                self.uniques.append(int(uniques[k]))
                self.counts.append(count)
                count=0    
        else:
            if classif_mode==0 and select_all==0:
                for k in range(len(uniques)):
                    for i in range(self.temp.shape[0]):
                        if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[k]:
                            if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temp.iloc[i,:]])
                                temps_c=temps_c.reshape((1,temps_c.shape[1]))
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                    self.uniques.append(int(uniques[k]))
                    self.counts.append(count)
                    count=0   
            else:
                if classif_mode==1 and select_all==1:
                    for i in range(max_length):
                        if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[class_label]:
                            if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temp.iloc[i,:]])
                                temps_c=temps_c.reshape((1,temps_c.shape[1]))
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                        if i<self.temps.shape[0] and self.temps.iloc[i,0]==uniques[class_label]:
                            if npy.isnan(npy.sum(self.temps.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temps.iloc[i,:]]).reshape(1,self.temps.shape[1])
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                    self.uniques.append(int(uniques[class_label]))
                    self.counts.append(count)
                    count=0
                else:
                    for i in range(self.temp.shape[0]):
                        if i<self.temp.shape[0] and self.temp.iloc[i,0]==uniques[class_label]:
                            if npy.isnan(npy.sum(self.temp.iloc[i,1:])):
                                 self.rejected.append(i)
                            else:
                                count=count+1
                                temps_c=npy.array([self.temp.iloc[i,:]])
                                temps_c=temps_c.reshape((1,temps_c.shape[1]))
                                self.data=npy.concatenate((self.data,temps_c),axis=0)
                    self.uniques.append(int(uniques[class_label]))
                    self.counts.append(count)
                    count=0  
        self.final_estimate=npy.zeros((len(self.uniques),self.data.shape[1]-1))
        self.data=self.data[1:,:]
        data=npy.zeros((1,self.data.shape[1]))
        self.my_uniqes_copy=npy.copy(self.uniques)
        self.validation_data=npy.zeros((1,self.data.shape[1]))
        for i in range(len(self.uniques)):
            temping=self.data[self.data[:,0]==self.uniques[i]]
            temping[:,0]=i
            data=npy.concatenate((data,temping))
            self.uniques[i]=i
        self.validation=validation
        self.data=data[1:,:]
        print('new data labels',self.data[:,0])
        self.copy_of_data=npy.copy(self.data)
        self.my_loss_count=0
        calc_val=int(self.temp.shape[0]*self.validation)
        print('validation',calc_val)
        if calc_val>0:
            count=0
            if calc_val>=len(self.uniques):
                for i in range(len(self.uniques)):
                    data=self.data[self.data[:,0]==i]
                    vals=int(data.shape[0]*self.validation)
                    if vals==0 and data.shape[0]>1:
                            vals=1
                    if vals>=1:
                        for j in range(vals):
                            data=self.data[self.data[:,0]==i]
                            ind=npy.random.randint(0,data.shape[0]-1)
                            self.validation_data=npy.concatenate((self.validation_data,data[ind,:].reshape(1,data.shape[1])),axis=0)
                            res=npy.where(npy.all(self.data==data[ind,:],axis=1))
                            data=npy.delete(self.data,ind,0)
                            self.data=npy.delete(self.data,res[0][0],0)
                self.validation_data=self.validation_data[1:,:]
            else:
                npy.random.shuffle(self.data)
                self.validation_data=self.data[self.data.shape[0]-calc_val:,:]
                self.data=self.data[0:self.data.shape[0]-calc_val,:]
        print('My validation classes:',self.validation_data[:,0])
        npy.random.shuffle(self.data)
        concat_temp=npy.concatenate((self.data,self.validation_data),axis=0)
        lab=concat_temp[:,0].reshape((concat_temp.shape[0],1))
        self.my_normalized=StandardScaler().fit_transform(concat_temp[:,1:])
        self.my_normalized=npy.concatenate((lab,concat_temp),axis=1)
        self.my_normalized=concat_temp[0:self.data.shape[0],:]
        self.my_normalized_valid=concat_temp[self.data.shape[0]:,:]
        self.confusion_matrix=npy.zeros((len(self.uniques),len(self.uniques)))
        print('Total Data sets in train file:',self.temp.shape[0])
        print('Total Data set in test file:',self.temps.shape[0])
        print('Totoal Unique classes in the data sets:',len(uniques))
        print('Unique Class labels:',uniques)
        print('selected unique class for the experiment:',self.uniques)
        print('Data sets in the selected unique class:',self.counts)
        print('Selected Centroid Mode:',self.classif_mode)
        print('Data sets rejected due to NA',len(self.rejected))
        print('Data size selected for training:',self.data.shape[0])
        print('Data size selected for validation',self.validation_data.shape[0])
        print('Normalized data',self.my_normalized.shape)
        print('Normalized validation data',self.my_normalized_valid.shape)
    def Build_Inception_Multitask(self,quantiles,encoder_node_size,filter_size, polling_size,En_L1_reg,En_L2_reg, De_L1_reg,De_L2_reg,Cl_L1_reg,Cl_L2_reg,Input_activ, Hidden_activ, Learning_rate):
        initializer=tf.keras.initializers.he_normal()
        initializer2 = tf.keras.initializers.GlorotNormal()
        En_inputs=Input(shape=(self.data.shape[1]-1,))
        self.Encoder_layer=Sequential()
        self.encoder_node_size=encoder_node_size
        #self.comps=components
        Inps=layers.Reshape((self.data.shape[1]-1,1), input_shape=(self.data.shape[1]-1,),name='EL1')(En_inputs)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Input_activ,name='IP_L1')(Inps)
        Inps1=layers.Conv1D(32,5, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL2')(pass_inps)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L2')(Inps)
        Inps2=layers.Conv1D(32,3, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL3')(pass_inps)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L3')(Inps)
        f_2filter=layers.Conv1D(32,2, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL4')(pass_inps)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L4')(Inps)
        concats=layers.concatenate([Inps1,Inps2,f_2filter,pass_inps],name='Concat_L1',axis=2)
        max_pooling_enc=layers.MaxPooling1D(polling_size,padding='same',name='Max_L1')(concats)
        
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L5')(max_pooling_enc)
        Inps1=layers.Conv1D(16,5, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL5')(pass_inps)
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L6')(max_pooling_enc)
        Inps2=layers.Conv1D(16,3, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL6')(pass_inps)
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L7')(max_pooling_enc)
        f_2filter=layers.Conv1D(16,2, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL7')(pass_inps)
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L8')(max_pooling_enc)
        concats=layers.concatenate([Inps1,Inps2,f_2filter,pass_inps],name='Concat_L2',axis=2)
        max_pooling_enc=layers.MaxPooling1D(polling_size,padding='same',name='Max_L2')(concats)
        
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L9')(max_pooling_enc)
        Inps1=layers.Conv1D(8,5, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL8')(pass_inps)
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L10')(max_pooling_enc)
        Inps2=layers.Conv1D(8,3, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL9')(pass_inps)
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L11')(max_pooling_enc)
        f_2filter=layers.Conv1D(8,2, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL10')(pass_inps)
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='IP_L12')(max_pooling_enc)
        concats=layers.concatenate([Inps1,Inps2,f_2filter,pass_inps],name='Concat_L3',axis=2)
        max_pooling_enc=layers.MaxPooling1D(strides=1,padding='same',name='Max_L3')(concats)
        
        E_flaten=layers.Flatten(name='EL14')(max_pooling_enc)
        E_out=layers.Dense(int(encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='features_output')(E_flaten)
        self.Encoder_layer=Model(En_inputs,E_out)
        self.Encoder_layer.summary()
        Dec_inps=Input(shape=(self.encoder_node_size,))
        Dec_Inp=layers.Reshape((encoder_node_size,1),input_shape=(encoder_node_size,),name='DL1')(Dec_inps)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L1')(Dec_Inp)
        Inps1=layers.Conv1D(32,5, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL2')(pass_inps)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L2')(Dec_Inp)
        Inps2=layers.Conv1D(32,3, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL3')(pass_inps)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L3')(Dec_Inp)
        f_2filter=layers.Conv1D(32,2, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DL4')(pass_inps)
        pass_inps=layers.Conv1D(32,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L4')(Dec_Inp)
        concats=layers.concatenate([Inps1,Inps2,f_2filter,pass_inps],name='D_Concat_L1',axis=2)
        concats=layers.Conv1DTranspose(128,filter_size,strides=2,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='up_sample1')(concats)
        
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L5')(concats)
        Inps1=layers.Conv1D(16,5, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL5')(pass_inps)
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L6')(concats)
        Inps2=layers.Conv1D(16,3, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL6')(pass_inps)
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L7')(concats)
        f_2filter=layers.Conv1D(16,2, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DL7')(pass_inps)
        pass_inps=layers.Conv1D(16,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L8')(concats)
        concats=layers.concatenate([Inps1,Inps2,f_2filter,pass_inps],name='D_Concat_L2',axis=2)
        concats=layers.Conv1DTranspose(64,filter_size,strides=2,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='up_sample2')(concats)
        
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L9')(concats)
        Inps1=layers.Conv1D(8,5, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL9')(pass_inps)
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L10')(concats)
        Inps2=layers.Conv1D(8,3, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL10')(pass_inps)
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L11')(concats)
        f_2filter=layers.Conv1D(8,2, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DL11')(pass_inps)
        pass_inps=layers.Conv1D(8,1, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='DP_L12')(concats)
        concats=layers.concatenate([Inps1,Inps2,f_2filter,pass_inps],name='D_Concat_L3',axis=2)
        concats=layers.Conv1DTranspose(32,filter_size,strides=1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='up_sample3')(concats)
        D_flatten=layers.Flatten(name='DL13')(concats)
        D_out=layers.Dense(self.data.shape[1]-1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Input_activ,name="Decoder_output")(D_flatten)
        self.Decoder_layer=Model(Dec_inps,D_out)
        self.Decoder_layer.summary()
        classifier_input=Input(shape=(self.encoder_node_size,))
        self.Classifier_layer=layers.Dense(int(encoder_node_size-0.1*self.encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='CL2')(classifier_input)
        self.Classifier_layer=layers.Dense(int(encoder_node_size-0.2*self.encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='CL3')(self. Classifier_layer)
        cl_out=layers.Dense(int(len(self.uniques)),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation='softmax',kernel_initializer=initializer2,name="Classifier_output")(self.Classifier_layer)
        self.Classifier_layer=Model(classifier_input,cl_out)
        self.Classifier_layer.summary()
        self.mymodel=trainer.Custom_trainer(self.Encoder_layer,self.Classifier_layer,self.Decoder_layer,len(self.uniques),quantiles)
        self.mymodel.compile(optimizer=optimizers.Adam(lr=Learning_rate))
    def Multi_task_encoder_two(self,quantiles,encoder_node_size,filter_size, polling_size,En_L1_reg,En_L2_reg, De_L1_reg,De_L2_reg,Cl_L1_reg,Cl_L2_reg,Input_activ, Hidden_activ, Learning_rate):
        initializer=tf.keras.initializers.he_normal()
        initializer2 = tf.keras.initializers.GlorotNormal()
        En_inputs=Input(shape=(self.data.shape[1]-1,))
        self.Encoder_layer=Sequential()
        self.encoder_node_size=encoder_node_size
        self.Encoder_layer=layers.Reshape((self.data.shape[1]-1,1),input_shape=(self.data.shape[1]-1,),name='EL1')(En_inputs)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,name='EL2')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL3')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL4')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL5',padding='same')(self.Encoder_layer)
        
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',kernel_initializer=initializer,name='EL6')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='EL7')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',kernel_initializer=initializer,name='EL8')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,name='EL9',padding='same')(self.Encoder_layer)
        
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',kernel_initializer=initializer,name='EL10')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='EL11')(self.Encoder_layer)
        self.Encoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ, padding='same',kernel_initializer=initializer,name='EL12')(self.Encoder_layer)
        self.Encoder_layer=layers.MaxPooling1D(polling_size,strides=1,padding='same',name='EL13')(self.Encoder_layer)
        
        E_out1=layers.Flatten(name='EL14')(self.Encoder_layer)
        E_out=layers.Dense(encoder_node_size,kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='Encoder_output')(E_out1)
        self.Encoder_layer=Model(En_inputs,E_out,name="encoder")
        self.Encoder_layer.summary()
        
        Dec_input=Input(shape=(self.encoder_node_size,))
        self.Decoder_layer=layers.Reshape((encoder_node_size,1),input_shape=(encoder_node_size,),name='DL1')(Dec_input)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL2')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL3')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL4')(self.Decoder_layer)
        
        self.Decoder_layer=layers.Conv1DTranspose(128,filter_size,strides=2,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL5')(self.Decoder_layer)
        
        
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL6')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL7')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL8')(self.Decoder_layer)
        
        self.Decoder_layer=layers.Conv1DTranspose(64,filter_size,strides=2,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL9')(self.Decoder_layer)
        
        
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL10')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL11')(self.Decoder_layer)
        self.Decoder_layer=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL12')(self.Decoder_layer)
        
        self.Decoder_layer=layers.Conv1DTranspose(32,filter_size,strides=1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='DL13')(self.Decoder_layer)
        
        self.Decoder_layer=layers.Flatten(name='DL14')((self.Decoder_layer))
        D_out=layers.Dense(self.data.shape[1]-1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Input_activ,name="Decoder_output")(self.Decoder_layer)
        self.Decoder_layer=Model(Dec_input,D_out,name="decoder")
        self.Decoder_layer.summary()
        
        classifier_input=Input((self.encoder_node_size,))
        self.Classifier_layer=layers.Dense(int(encoder_node_size-0.1*self.encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='CL2')(classifier_input)
        self.Classifier_layer=layers.Dense(int(encoder_node_size-0.2*self.encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='CL3')(self. Classifier_layer)
        cl_out=layers.Dense(int(len(self.uniques)),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation='softmax',kernel_initializer=initializer2,name="Classifier_output")(self. Classifier_layer)
        self.Classifier_layer=Model(classifier_input,cl_out)
        self.Classifier_layer.summary()
        self.mymodel=trainer.Custom_trainer(self.Encoder_layer,self.Classifier_layer,self.Decoder_layer,len(self.uniques),quantiles)
        self.mymodel.compile(optimizer=optimizers.Adam(lr=Learning_rate))
    def Build_VGG_ResNet_Multitasking(self,quantiles,encoder_node_size,filter_size, polling_size,En_L1_reg,En_L2_reg,De_L1_reg,De_L2_reg,Cl_L1_reg,Cl_L2_reg,Input_activ, Hidden_activ,Learning_rate): 
        En_inputs=Input(shape=(self.data.shape[1]-1,))
        self.encoder_node_size=encoder_node_size
        initializer=tf.keras.initializers.he_normal()
        initializer2 = tf.keras.initializers.GlorotNormal()
        Inps=layers.Reshape((self.data.shape[1]-1,1), input_shape=(self.data.shape[1]-1,),name='EL1')(En_inputs)
        Inps1=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Input_activ,name='EL2')(Inps)
        Inps2=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL3')(Inps1)
        Inps2=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL4')(Inps2)
        Inps3=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),padding='same',activation=Hidden_activ,kernel_initializer=initializer,name='EL5')(Inps2)
        adds=layers.Add(name='ADD_L1')([Inps3,Inps])
        adds_max=layers.MaxPooling1D(polling_size,padding='same',name='ADD_Max_L1')(adds)
        Inps1=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='EL6')(adds_max)
        Inps2=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='EL7')(Inps1)
        Inps2=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='EL8')(Inps2)
        Inps3=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='EL9')(Inps2)
        adds=layers.Add(name='ADD_L2')([Inps3,adds_max])
        adds_max=layers.MaxPooling1D(polling_size,padding='same',name='ADD_Max_L2')(adds)
        Inps1=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='EL10')(adds_max)
        Inps2=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='EL11')(Inps1)
        Inps3=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='EL12')(Inps2)
        adds=layers.Add(name='ADD_L3')([Inps3,adds_max])
        adds_max=layers.MaxPooling1D(polling_size,strides=1,padding='same',name='ADD_Max_L3')(adds)
        E_flaten=layers.Flatten(name='EL13')(adds_max)
        E_out=layers.Dense(int(encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=En_L1_reg,l2=En_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='features_output')(E_flaten)
        self.Encoder_layer=Model(En_inputs,E_out)
        self.Encoder_layer.summary()
        
        Dec_input=Input(shape=(self.encoder_node_size,))
        Dec_Inp=layers.Reshape((encoder_node_size,1),input_shape=(encoder_node_size,),name='DL1')(Dec_input)
        Inps1=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL2')(Dec_Inp)
        Inps2=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL3')(Inps1)
        Inps2=layers.Conv1D(128,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL4')(Inps2)
        Inps3=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL5')(Inps2)
        adds=layers.Add(name='ADD_L4')([Inps3,Dec_Inp])
        adds_max=layers.Conv1DTranspose(32,filter_size,strides=2,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='ADD_Max_L4')(adds)
        
        Inps1=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL6')(adds_max)
        Inps2=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL7')(Inps1)
        Inps2=layers.Conv1D(64,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL8')(Inps2)
        Inps3=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL9')(Inps2)
        adds=layers.Add(name='ADD_L5')([Inps3,adds_max])
        adds_max=layers.Conv1DTranspose(32,filter_size,strides=2,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='ADD_Max_L5')(adds)
        Inps1=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL10')(adds_max)
        Inps2=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL11')(Inps1)
        Inps3=layers.Conv1D(32,filter_size, kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,padding='same',name='DL12')(Inps2)
        adds=layers.Add(name='ADD_L6')([Inps3,adds_max])
        adds_max=layers.Conv1DTranspose(32,filter_size,strides=1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Hidden_activ,padding='same',kernel_initializer=initializer,name='ADD_Max_L6')(adds)
        D_flatten=layers.Flatten(name='DL13')(adds_max)
        D_out=layers.Dense(self.data.shape[1]-1,kernel_regularizer=regularizers.l1_l2(l1=De_L1_reg,l2=De_L2_reg),activation=Input_activ,name="Decoder_output")(D_flatten)
        self.Decoder_layer=Model(Dec_input,D_out,name="decoder")
        self.Decoder_layer.summary()
        
        classifier_input=Input((self.encoder_node_size,))
        self.Classifier_layer=layers.Dense(int(encoder_node_size-0.1*self.encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='CL2')(classifier_input)
        self.Classifier_layer=layers.Dense(int(encoder_node_size-0.2*self.encoder_node_size),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation=Hidden_activ,kernel_initializer=initializer,name='CL3')(self. Classifier_layer)
        cl_out=layers.Dense(int(len(self.uniques)),kernel_regularizer=regularizers.l1_l2(l1=Cl_L1_reg,l2=Cl_L2_reg),activation='softmax',kernel_initializer=initializer2,name="Classifier_output")(self.Classifier_layer)
        self.classifier_layer=Model(classifier_input,cl_out)
        self.classifier_layer.summary()
        self.mymodel=trainer.Custom_trainer(self.Encoder_layer,self.classifier_layer,self.Decoder_layer,len(self.uniques),quantiles)
        self.mymodel.compile(optimizer=optimizers.Adam(lr=Learning_rate))
    def estimate_mean_separately(self):
        my_test=npy.array(self.temps.iloc[:,:])
        my_uniques=npy.unique(my_test[:,0])
        data=npy.zeros((1,my_test.shape[1]))
        for i in range(len(my_uniques)):
            temp=my_test[my_test[:,0]==my_uniques[i]]
            temp[:,0]=i
            data=npy.concatenate((data,temp))
        my_test=data[1:,:]
        my_lat_rep=npy.zeros((1,self.encoder_node_size+1))
        my_uniques=self.uniques
        print(my_test.shape)
        print(my_uniques)
        est_mean=npy.zeros((len(my_uniques),self.data.shape[1]-1))
        my_lat_mean=npy.zeros((len(my_uniques),self.encoder_node_size))
        for k in range(len(my_uniques)):
            temp=my_test[my_test[:,0]==my_uniques[k]]
            print(temp.shape)
            lat_predict=self.Encoder_layer.predict(temp[:,1:])
            class_lab=temp[:,0].reshape((temp.shape[0],1))
            lat_mean=npy.mean(lat_predict,0)
            my_lat_mean[k,:]=lat_mean.reshape((1,self.encoder_node_size))
            lat_predict=npy.concatenate((class_lab,lat_predict),axis=1)
            my_lat_rep=npy.concatenate((my_lat_rep,lat_predict),axis=0)
            print('latent_mean',lat_mean.shape)
            est_mean[k,:]=self.Decoder_layer.predict(npy.array([lat_mean]))
        encoded=self.Encoder_layer.predict(my_test[:,1:])
        decoded=self.Decoder_layer.predict(encoded)
        decoded=npy.concatenate((npy.reshape(my_test[:,0],(my_test.shape[0],1)),decoded),axis=1)
        return my_lat_rep[1:,:],my_lat_mean,my_test,decoded
    
    def fit_Gausian_Latent_two(self):
        my_lat_means=npy.zeros((1,self.encoder_node_size))
        latent_space_WGSS=npy.zeros((2,len(self.uniques)))
        Time_Domain_WGSS=npy.zeros((2,len(self.uniques)))
        concat=npy.concatenate((self.data,self.validation_data),axis=0)
        my_latents=[]
        for k in range(len(self.uniques)):
            data=concat[concat[:,0]==self.uniques[k]]
            data=data.reshape((data.shape[0],self.data.shape[1]))
            latents=self.Encoder_layer.predict(data[:,1:])
            latent_means=npy.mean(latents,axis=0)
            latent_means=latent_means.reshape((1,self.encoder_node_size))
            my_lat_means=npy.concatenate((my_lat_means,latent_means),axis=0)
            labels=data[:,0].reshape((data.shape[0],1))
            latent_space_WGSS[0,k]=npy.mean(npy.sqrt(npy.sum(npy.square(latents-latent_means),axis=1)))
            latent_space_WGSS[1,k]=metrics.dtw(latents,latent_means)/data.shape[0]
            Time_Domain_WGSS[0,k]=npy.mean(npy.sqrt(npy.sum(npy.square(data[:,1:]-self.Decoder_layer.predict(latent_means)),axis=1)))
            Time_Domain_WGSS[1,k]=metrics.dtw(data[:,1:],self.Decoder_layer.predict(latent_means))/data.shape[0]
            if k==0:
                    my_latents=npy.concatenate((labels,latents),axis=1)
            else:
                    append_lab=npy.concatenate((labels,latents),axis=1)
                    my_latents=npy.concatenate((my_latents,append_lab),axis=0)
        my_lat_means=my_lat_means[1:,:]
        print('Estimated means',my_lat_means)
        self.final_estimate=self.Decoder_layer.predict(my_lat_means)
        return my_latents,my_lat_means,self.final_estimate,latent_space_WGSS,Time_Domain_WGSS 
    
    def load_model_weights(self,model_name,data_path='D:\\Deep learning data\\Averaging via autoencoder\\'):
        model_path=data_path.replace('\\','/')+model_name#.replace('.ckpt','')
        train_hist_path=data_path.replace('\\','/')
        path=train_hist_path+model_name.replace('vae','')+'train_hsitoryconv '+'.txt'
        if os.path.isfile(path):
            with(open(path, 'rb')) as file_pi:
                historys=pickle.load(file_pi) 
            self.mymodel.load_weights(model_path)
            self.hist_dict=historys
        else:
            print('Model weights not found! Check model path and name. ')

    def Train_Warper_Model(self,user_epochs,bathc_size_factor,unique_id=''):
         checkpath=self.Model_save_path+self.Model_save_name+str(unique_id)
         cp_store=ModelCheckpoint(checkpath,save_weights_only=True,monitor='val_total_loss',save_best_only=True,verbose=1)
         start_time=time.time()
         calc_valid=int(self.temp.shape[0]*self.validation)
         historys=''
         if calc_valid==0: 
             labels=npy.reshape(self.data[:,0],(self.data.shape[0],1))
             catagorical_labels=to_categorical(labels,num_classes=len(self.uniques))
             historys=self.mymodel.fit(x=self.data[:,1:],y=(self.data[:,1:],catagorical_labels,labels),epochs=user_epochs,verbose=1,batch_size=int((self.data.shape[0])/bathc_size_factor))
             with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
                 pickle.dump(historys.history, file_pi)
             self.hist_dict=historys.history
             self.training_time=time.time()-start_time
             #model_paths=checkpath.replace('.ckpt','')
             #self.mymodel.save_weights(model_paths+'.h5')
         else:
             labels=npy.reshape(self.data[:,0],(self.data.shape[0],))
             val_labels=npy.reshape(self.validation_data[:,0],(self.validation_data.shape[0],))
             cat_train_labels=to_categorical(labels.reshape(self.data.shape[0],1),num_classes=len(self.uniques))
             cat_val_labels=to_categorical(val_labels.reshape(self.validation_data.shape[0],1),num_classes=len(self.uniques))
             historys=self.mymodel.fit(x=self.data[:,1:],y=(self.data[:,1:],cat_train_labels,labels),validation_data=[self.validation_data[:,1:],self.validation_data[:,1:],cat_val_labels,val_labels],epochs=user_epochs,verbose=1,batch_size=int((self.data.shape[0])/bathc_size_factor))
             with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
                 pickle.dump(historys.history, file_pi)
             self.hist_dict=historys.history
             self.training_time=time.time()-start_time
             #model_paths=checkpath.replace('.ckpt','')
             #self.mymodel.save_weights(model_paths+'.h5')
         with(open(checkpath+'train_hsitoryconv '+'.txt', 'wb')) as file_pi:
             pickle.dump(historys.history, file_pi)
         self.hist_dict=historys.history
         self.training_time=time.time()-start_time
         if os.path.isfile(checkpath+' Comment.txt'):
             with(open(checkpath+' Comment.txt', 'a')) as file_pi:
                 file_pi.write('Time taken to train the network:'+str(self.training_time))
         else:
              with(open(checkpath+' Comment.txt', 'w')) as file_pi:
                 file_pi.write('Time taken to train the network:'+str(self.training_time))
         print('Time taken for training in seconds: ', self.training_time)
         if os.path.isfile(checkpath+' losses.txt'):
             leng=len(self.hist_dict['total_loss'])
             with(open(checkpath+' losses.txt', 'a')) as file_pi:
                 if self.validation>0:
                     file_pi.write('Final Validation Loss'+str(self.hist_dict['val_total_loss'][leng-1])+'Final training loss'+str(self.hist_dict['total_loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['rec_loss'][leng-1])+'Final Decoder val_loss'+str(self.hist_dict['val_rec_loss'][leng-1])+'Final registration val_loss'+str(self.hist_dict['val_reg_loss'][leng-1])+'Final registration loss'+str(self.hist_dict['reg_loss'][leng-1]))
                 else:
                     file_pi.write('Final training loss'+str(self.hist_dict['total_loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['rec_loss'][leng-1])+'Final registration loss'+str(self.hist_dict['reg_loss'][leng-1]))
         else:
              leng=len(self.hist_dict['total_loss'])
              with(open(checkpath+' losses.txt', 'w')) as file_pi:
                 if self.validation>0:
                     file_pi.write('Final Validation Loss'+str(self.hist_dict['val_total_loss'][leng-1])+'Final training loss'+str(self.hist_dict['total_loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['rec_loss'][leng-1])+'Final Decoder val_loss'+str(self.hist_dict['val_rec_loss'][leng-1])+'Final registration val_loss'+str(self.hist_dict['val_reg_loss'][leng-1])+'Final registration loss'+str(self.hist_dict['reg_loss'][leng-1]))
                 else:
                     file_pi.write('Final training loss'+str(self.hist_dict['total_loss'][leng-1])+'Final Decoder loss'+str(self.hist_dict['rec_loss'][leng-1])+'Final registration loss'+str(self.hist_dict['reg_loss'][leng-1]))
    def write_means(self,WGSS_mean=1,res=0,save_name=' estimated_means.tsv'):
        means=self.final_estimate
        Filename=self.Model_save_path+self.File_name+save_name
        if os.path.isfile(Filename):
             with open(Filename, 'a', newline='') as tsv_file:
                print('opened file in',Filename )
                tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                if WGSS_mean==1:
                    for i in range(means.shape[0]):
                        tsv_writer.writerow(self.final_estimate[i,:])
                else:
                    tsv_writer.writerow([res])
        else:
            with open(Filename, 'w', newline='') as tsv_file:
                print('opened file in',Filename )
                tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                if WGSS_mean==1:
                    for i in range(means.shape[0]):
                        tsv_writer.writerow(self.final_estimate[i,:])
                else:
                    tsv_writer.writerow([res])
    def write_atsv_file(self,array,file_loc,file_name):
        Filename=file_loc+file_name
        for i in range(array.shape[0]):
            with open(Filename, 'a', newline='') as tsv_file:
                  #print('opened file in',Filename )
                  tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
                  tsv_writer.writerow(array[i,:])
        print('Finished Writting a file '+file_loc+file_name)