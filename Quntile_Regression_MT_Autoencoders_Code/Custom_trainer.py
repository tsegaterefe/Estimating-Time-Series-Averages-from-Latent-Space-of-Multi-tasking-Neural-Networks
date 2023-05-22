# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 14:43:52 2021

@author: Tsega
"""
import keras 
import tensorflow as tf
import tensorflow.keras.backend as K
class Custom_trainer(tf.keras.Model):
    def __init__(self,encoder,classifier,decoder,uniques,epsilon,**kwargs):
        super(Custom_trainer, self).__init__(**kwargs)
        self.decoder=decoder
        self.encoder=encoder
        self.classifier=classifier
        self.uniques=uniques
        self.total_loss_tracker=tf.keras.metrics.Mean(name='total_loss')
        self.reg_loss_tracker=tf.keras.metrics.Mean(name='reg_loss')
        self.reconstruction_loss_tracker=tf.keras.metrics.Mean(name="rec_loss")
        self.classification_loss_tracker=tf.keras.metrics.Mean(name="class_loss")
        self.lat_rec_loss=tf.keras.metrics.Mean(name='lat_rec_loss')
        self.accuracy_metrics=tf.keras.metrics.CategoricalAccuracy(name='accuracy')
        self.epsilon=epsilon
    @property
    def metrics(self):
        return [self.reg_loss_tracker,self.lat_rec_loss,self.reconstruction_loss_tracker,self.classification_loss_tracker,self.total_loss_tracker,self.accuracy_metrics]

    @tf.function
    def train_step(self, data):
        with tf.GradientTape(persistent=True) as tape:
             inputs,prediction=data
             output,cat_labels,labels=prediction
             class_loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
             labels=tf.cast(labels,tf.int32)      
             encoded=self.encoder(inputs,training=True)
             decoded=self.decoder(encoded, training=True)
             classs_estimates=self.classifier(encoded,training=True)
             rec_latents=self.encoder(decoded,training=True)
             classes, class_idx = tf.unique(labels)
             for i in range(self.uniques):
                 selected_latents=tf.gather(encoded, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 selected_decoded=tf.gather(decoded, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 selected_output=tf.gather(output, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 selected_rec_lat=tf.gather(rec_latents, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 if i==0:
                    rec_loss=selected_decoded-selected_output
                    rec_lat_loss=selected_rec_lat-selected_latents
                    means=K.mean(selected_latents,axis=0)
                    wgss_loss=K.square(selected_latents-means)
                    
                 else:
                    temp_rec_lat_loss=selected_rec_lat-selected_latents
                    temp_rec_loss=selected_decoded-selected_output
                    means=K.mean(selected_latents,axis=0)
                    temp_wgss_loss=K.square(selected_latents-means)
                    wgss_loss=tf.concat((wgss_loss,temp_wgss_loss),axis=0)
                    rec_loss=tf.concat((temp_rec_loss,rec_loss),axis=0)
                    rec_lat_loss=tf.concat((temp_rec_lat_loss,rec_lat_loss),axis=0)
             estimator_time=[]
             estimator_latent=[]
             for i in range(len(self.epsilon)):
                      estimator_time.append(K.maximum(self.epsilon[i]*(rec_loss), (self.epsilon[i]-1)*(rec_loss)))
                      estimator_latent.append(K.maximum(self.epsilon[i]*(rec_lat_loss), (self.epsilon[i]-1)*(rec_lat_loss)))
             rec_loss=tf.reduce_mean(K.maximum(estimator_time[0],estimator_time[1]),axis=0)
             rec_latent_loss=tf.reduce_mean(K.maximum(estimator_latent[0],estimator_latent[1]),axis=0)
             cat_class_loss=class_loss(classs_estimates,cat_labels)
             total_loss=tf.reduce_mean(rec_loss,axis=1)+tf.reduce_mean(rec_latent_loss,axis=1)+tf.reduce_mean(wgss_loss,axis=1)+tf.reduce_mean(cat_class_loss,axis=0)
             enc_losses=tf.reduce_mean(wgss_loss,axis=1)+tf.reduce_mean(rec_loss,axis=1)+tf.reduce_mean(cat_class_loss,axis=0)
             dec_losses=tf.reduce_mean(rec_loss,axis=1)+tf.reduce_mean(rec_latent_loss,axis=1)
        enc_loss=tape.gradient(enc_losses,self.encoder.trainable_weights)
        dec_loss=tape.gradient(dec_losses,self.decoder.trainable_weights)
        class_loss=tape.gradient(cat_class_loss,self.classifier.trainable_weights)
        self.optimizer.apply_gradients(zip(enc_loss,self.encoder.trainable_weights))
        self.optimizer.apply_gradients(zip(dec_loss,self.decoder.trainable_weights))
        self.optimizer.apply_gradients(zip(class_loss,self.classifier.trainable_weights))
        self.reg_loss_tracker.update_state(wgss_loss)
        self.lat_rec_loss.update_state(rec_latent_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.classification_loss_tracker.update_state(cat_class_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_metrics.update_state(cat_labels,classs_estimates)
        return {
            'reg_loss':self.reg_loss_tracker.result(),'lat_rec_loss':self.lat_rec_loss.result(),'rec_loss': self.reconstruction_loss_tracker.result(),'class_loss':self.classification_loss_tracker.result(),'total_loss':self.total_loss_tracker.result(),'Accuracy':self.accuracy_metrics.result()
        }
    @tf.function
    def test_step(self,data):
        if isinstance(data, tuple):
            inputs= data[0][0]
            output= data[0][1]
            cat_labels=data[0][2]
            labels=data[0][3]
        labels=tf.cast(labels,tf.int32)
        encoded=self.encoder(inputs,training=False)
        decoded=self.decoder(encoded, training=False)
        classs_estimates=self.classifier(encoded,training=False)
        rec_latents=self.encoder(decoded,training=False)
        class_loss=tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        classes, class_idx = tf.unique(labels)
        for i in range(self.uniques):
                 selected_latents=tf.gather(encoded, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 selected_decoded=tf.gather(decoded, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 selected_output=tf.gather(output, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 selected_rec_lat=tf.gather(rec_latents, tf.where(tf.equal(class_idx, tf.cast(i, tf.int32))))
                 selected_latents=tf.cast(selected_latents,tf.float32)
                 selected_output=tf.cast(selected_output,tf.float32)
                 selected_decoded=tf.cast(selected_decoded,tf.float32)
                 if i==0:
                    rec_loss=selected_decoded-selected_output
                    rec_lat_loss=selected_rec_lat-selected_latents
                    _,wgss_loss=tf.nn.moments(selected_latents,axes=0)

                 else:
                    temp_rec_lat_loss=selected_rec_lat-selected_latents
                    temp_rec_loss=selected_decoded-selected_output
                    _,temp_wgss_loss=tf.nn.moments(selected_latents,axes=0)
                    wgss_loss=tf.concat((wgss_loss,temp_wgss_loss),axis=0)
                    rec_loss=tf.concat((temp_rec_loss,rec_loss),axis=0)
                    rec_lat_loss=tf.concat((temp_rec_lat_loss,rec_lat_loss),axis=0)
        cat_class_loss=class_loss(classs_estimates,cat_labels)
        estimator_time=[]
        estimator_latent=[]
        for i in range(len(self.epsilon)):
                      estimator_time.append(K.maximum(self.epsilon[i]*(rec_loss), (self.epsilon[i]-1)*(rec_loss)))
                      estimator_latent.append(K.maximum(self.epsilon[i]*(rec_lat_loss), (self.epsilon[i]-1)*(rec_lat_loss)))
             
        rec_loss=tf.reduce_mean(K.maximum(estimator_time[0],estimator_time[1]),axis=0)
        rec_latent_loss=tf.reduce_mean(K.maximum(estimator_latent[0],estimator_latent[1]),axis=0)
        total_loss=tf.reduce_mean(rec_loss,axis=1)+tf.reduce_mean(rec_latent_loss,axis=1)+tf.reduce_mean(wgss_loss,axis=1)+tf.reduce_mean(cat_class_loss,axis=0)
        self.reg_loss_tracker.update_state(wgss_loss)
        self.lat_rec_loss.update_state(rec_latent_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.classification_loss_tracker.update_state(cat_class_loss)
        self.total_loss_tracker.update_state(total_loss)
        self.accuracy_metrics.update_state(cat_labels,classs_estimates)
        return {
           'reg_loss':self.reg_loss_tracker.result(),'lat_rec_loss':self.lat_rec_loss.result(),'rec_loss': self.reconstruction_loss_tracker.result(),'class_loss':self.classification_loss_tracker.result(),'total_loss':self.total_loss_tracker.result(),'Accuracy':self.accuracy_metrics.result()
            }