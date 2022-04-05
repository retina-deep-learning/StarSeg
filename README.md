# StarSeg

Deep Learning for Multi-Label Segmentation of Optos Fundus Autofluorescence Images in Stargardt Disease 

This deep learning algorithm was trained to perform macular segmentation for grayscale Optos fundus autofluorescence images of patients with genetically confirmed Stargardt disease.



Prerequisites: 
Python 3.9.7 
Tensorflow 2.4.1 
Keras 2.6.0

Before starting, the model weights should be download and placed in 'src'. Download the model weights from:
https://github.com/retina-deep-learning/StarSeg/releases/download/v1.0.0/starseg_best_weights.h5

The code used for training and cross-validation can be viewed in 'run_train.py', 'lib/cnn_train.py', and 'lib/helper_functions.py'. Due to HIPAA regulations, the training data set is not publicly available. However, model weights may be re-trained or fine-tuned with other data sets.

The code used for prediction can be used to generate automated segmentations for images. Rename and format an Optos autofluorescence image to 'image.png' and replace the blank 'image.png' in 'input'. Execute the code using 'run_predict.py'. A 'results' folder will be generate with the following contents:

1. A copy of the original image file
2. A copy of the preprocessed macular image shown to the algorithm
3. The algorithm's automated macular segmentation
4. A text file containing the total segmented DDAF, QDAF, and OAAF areas


Acknowledgments

This work was supported by grants from the Foundation Fighting Blindness and National Eye Institute.


Disclaimer

The work described here is purely for research use only. The information obtained through use of these resources are not intended for diagnostic use or medical decision-making. The Foundation Fighting Blindness, National Eye Institute, and National Institutes of Health have not directly participated in the creation of this research tool, and have not reviewed or endorsed its validity or utility. The performance of this algorithm has not been evaluated by the Food and Drug Administration and is not intended for commercial or clinical use. If you have questions about this resource, please contact a local health care professional.

