# Estimating-Time-Series-Averages-from-Latent-Space-of-Multi-tasking-Neural-Networks
This repository contains Python scripts used for the pare "Estimating Time Series Averages from Latent Space of Multi-tasking Neural Networks" which is accepted at the Journal of Knowledge and Information System (KAIS)

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
 <img src="Images/Beetles_TS.png" height="250" width="500" >
 <img src="Images/Flies_TS.png" height="250" width="500" >
</p>
<p>
 <img src="Images/Beetles_Arth.png" height="250" width="500" >
 <img src="Images/Flies_Arth.png" height="250" width="500" >
</p>

# Proposed Architectures 
Overall, we proposed three multitasking autoencoders to estimate the averages of a range of temporal datasets extracted from the University of California Univariate Time Series Repository (UCR). The multitasking architectures were trained to minimize multi-class classification and reconstruction losses. Moreover, architecturewise, we have used layer arrangments that were utilized in the VGG16, ResNet, and Inception architectures.   
<p>
 <img src="Images/VGG_Based_MT_Arch.png" align="left" height="250" width="465" >
 <img src="Images/ResNet_Based_MT_Arch.png" align="right" height="250" width="470" >
 <p>
  
(a).&nbsp; Proposed VGG16 based Multi-tasking Autoencoder&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;  &nbsp; (b).&nbsp; Proposed ResNet based Multi-tasking Autoencoder
  </p>
</p>
<p align="center">
 <img src="Images/Inception_Based_MT_Arch.png" height="250" width="500" >
</p>
<p align="center">
  (c).&nbsp; Proposed Inception based Multi-tasking Autoencoder
 </p>

# Steps in Training the proposed Networks and Estimating Averages
