# Estimating-Time-Series-Averages-from-Latent-Space-of-Multi-tasking-Neural-Networks
This repository contains Python scripts used for the article *Estimating Time Series Averages from Latent Space of Multi-tasking Neural Networks* which is accepted at the journal of [ Knowledge and Information System (KAIS).](https://www.springer.com/journal/10115)

# Abstract
<p align="justify">
Time series averages are one key input to temporal data mining techniques such as classification, clustering, forecasting, etc. In practice, the optimality of estimated averages often impacts the performance 
of such temporal data mining techniques. Practically, an estimated average is presumed to be optimal if it minimizes the discrepancy between itself and members of an averaged set while preserving descriptive shapes. 
However, estimating an average under such constraints is often not trivial due to temporal shifts. To this end, all pioneering averaging techniques propose to align averaged series before estimating an average. 
Practically, the alignment gets performed to transform the averaged series, such that, after the transformation, they get registered to their arithmetic mean. However, in practice, most proposed alignment techniques 
often introduce additional challenges. For instance, Dynamic Time Warping~(DTW) based alignment techniques make the average estimation process non-smooth, non-convex, and computationally demanding. With such observation 
in mind, we approach time series averaging as a generative problem. Thus, we propose to mimic the effects of temporal alignment in the latent space of multi-tasking neural networks. We also propose to estimate~(augment) 
time domain averages from the latent space representations. With this approach, we provide state-of-the-art latent space registration. Moreover, we provide time domain estimations that are better than the estimates 
generated by some pioneering averaging techniques.

# Visual Demonstration of the Problem
<p>
 <img src="Images/Beetles_TS.png" height="250" width="400" >
 <img src="Images/Flies_TS.png" height="250" width="400" >
</p>
<p>
 <img src="Images/Beetles_Arth.png" height="250" width="400" >
 <img src="Images/Flies_Arth.png" height="250" width="400" >
</p>

# Proposed Architectures 
Overall, we proposed three multitasking autoencoders to estimate the averages of a range of temporal datasets obtained from the [University of California Univariate Time Series Repository (UCR).](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) The multitasking architectures were trained to minimize multi-class classification and reconstruction losses. Moreover, architecturewise, we have used layer arrangments that were utilized in the VGG16, ResNet, and Inception architectures.   
<p>
 <img src="Images/VGG_Based_MT_Arch.png" align="left" height="250" width="350" >
 <img src="Images/ResNet_Based_MT_Arch.png" align="right" height="250" width="350" >
 <br clear="right"/>
(a).&nbsp; Proposed VGG16 based Multi-tasking Autoencoder&nbsp; &nbsp; &nbsp; (b).&nbsp; Proposed ResNet based Multi-tasking Autoencoder
</p>
<p align="center">
 <img src="Images/Inception_Based_MT_Arch.png" height="250" width="350" >
</p>
<p align="center">
  (c).&nbsp; Proposed Inception based Multi-tasking Autoencoder
 </p>

# Steps in Training the Proposed Networks and Estimating Averages

* <p align="justify">The proposed architectures were first trained using the train splits of 114 datasets obtained from the UCR. For each datasets, the networks optimized for multi-class catagorical cross entropy, reconstruction loss, and time domain and latent space qantile regression losses. </p>
* <p align="justify">After training the proposed architecures on a given UCR dataset, the encoder portion of trained networks were then used to project the multi-class time domain dataset of a train split into the latent space.  </p>
* <p align="justify">After the projection, the per class atihimetic mean of the multi-class latent space embedding of the training split were taken as the estimate of the class averages. </p>
* <p align="justify">The per class latent space averages were then projected to the time domain using the decoder portion of the proposed trained multi-tasking autoencoders. </p>
* <p align="justify">Finally, to assess the quality of the estimated time domain per class averages, one nearest centroid classification (1NCC) was conducted using the estimated per class  latent space and time domain averages and the time domain and latent space embedding of test split of the UCR datasets.  </p>

# Sample Results

We have used different statistical assesment techniques to compare 1NCC accuracies that are obtained using averages estimated by different averaging techniques and our proposals. In the Figures shown below, we present some of the comparisions reported on the article. Beside the statistical comparisions, a visual demonstration of averages generated by our proposed apporaches for some [UCR](https://www.cs.ucr.edu/%7Eeamonn/time_series_data_2018/) datasets can be found [here](https://github.com/tsegaterefe/Sample-Averages-).    
<p align="center">
 <img src="Images/Box_Whisker_plot.png" align="center" height="250" width="700" >
</p>
<p align="center">
(a).&nbsp; Box Whisker plot comparision of 1NCC accuracies obtained on 84 UCR datasets.
 </p>
 <p align="center">
 <img src="Images/114_repeated_AGLVQ_Included_max_Time_Domain.png" align="center" height="250" width="700" >
</p>
<p align="center">
(b).&nbsp; Hypothesis test using on 1NCC accuracies obtaine on 114 UCR datasets.
 </p>
 
# How to Use the Provided Codes  
In the *Quntile_Regression_MT_Autoencoders_Code* folder, we have provided the Python implimentation for the Quantile regression multi-tasking autoencoders. Moreover, in the articel, we have also assessed a basic multi-tasking autoencoder that optimized for multi-class classificaiton and time domain reconstruction loss. The Python implimentation of this archtitectures was adopted from our privious work which could be found [here](https://github.com/tsegaterefe/Time-Series-Averaging-Using-Multi-Tasking-Autoencoder). In general, to utilize the provided scripts the following steps could be taken.
* <p align="justify">Copy the five Python Scripts and the the two .csv files to a folder, i.e., Dataset_List.csv and Startindex.csv.</p>
* <p align="justify">Dataset_List.csv contains the list of UCR datasets over which our porposed approaches were evaluated. Moreover, our porposed approachs were trained 25 times on a given UCR datasets to account for randomness in neural network parameters. The Startindex.csv keeps a record of how many times the networks were traiend on a given UCR dataset.</p>
* After copying a files, open the *Warping_Reg_Net_Config.py* script and specify the following parameters:  

  * self.File_loc to the directory where the UCR datasets are stored.  
  * self.Model_save_path to the directory the weights of the trained models get saved.
  * self.List_of_data_sets to the directory where *Dataset_List.csv* is stored. 
  * self.To_start_from to the the directory where *Startindex.csv* is stored.
  * self.run_per_single_data to the number of repeated training over a sigle UCR dataset (default is set to 25). A training session over a given UCR dataset ends when the a row in *Startindex.csv* that is associated with a dataset reaches this value.
  * self.Quantiles used to set &lambda; pair values that could encourage (discorage) over (under) estiamtinons at the decoder. By default, the &lambda; pair values are set to discourage over (under) estimations at the decoder. However, if the contrary is desired, the commented &lambda; pair values should be used. 
  * self.model_type to 1, 2, or 3. Setting it to 1 selects the VGG16 based quantile multi-tasking autoencoder. However, setting it to 2 or 3 respectively selects the Inception or ResNet based Quantile multi-tasking autoencoders.
  * self.First_time_train to 1 if the a selected model type is to be trained from scrathced. However, if it is to be loaded, set this parameter to 0 given the directory indicated by the *self.Model_save_path* has the proper weigth values. 
  * Finally, leave the remaining parameters to default values except the self.batch_size and self.Epoch which could be configured to a desired value. 
* After performing the necessary configurations, run the *Warping_Reg_Net_Main* script to start the training and estimation process. The *Warping_Reg_Net_Main* will create folders in the path indicated by the self.Model_save_path parameter for every UCR dataset listed in *Dataset_List.csv*. Morover, in the same path, an excel file is created to store the 1NCC accuracies for each UCR dataset and for each repeated training and estimation session associated with a given UCR dataset.

# Research Funding

 This research was conducted under the Ethio-France PhD. Program which was financed by:
 * The former Ethiopian Ministery of Science and Higher Education (MOSHE)
 * The French Embassy to Ethiopia and African Union.

We would like to acknowledge both parties for their generous contributions. 
