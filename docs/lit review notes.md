1. Exploratory Data Analysis of Image Texture and Statistical Features on Myocardium and Infarction Areas in Cardiac Magnetic Resonance Images  
    • Extracting CV-related features (mean pixel value, etc), statistical features (means, medians, sd) from areas of interest   
    • Extract “textual” features (texture of the region of interest) to measure properties such as smoothness, coarseness, and regularity. Described in three principal approaches: statistical (calculated by GLCM matrix), structural and spectral.  
    • In order to select features best describe and classify the data into two desired class, a Bayes classifier is trained, and Leave-one-out resampling is used. Bayesian is train with density function p(x|wi) and prior P(wi).  
2. 
3. Automatic classification of defective photovoltaic module cells in electroluminescence images
    • Propose two pipeline for determining a per-cell defect likelihood:  
        ◦ SVM – trained with various features extracted from images (less computation complexity)  
            ▪ Extract local descriptors (typically at salient points or from a dense pixel grid)  
            ▪ Compute a global representation and classify it into defective or functional class  
            ▪ Create binary masking, separating cells’ foreground from the background – strictly limit feature extraction to the cell interior  
            ▪ Extract feature descriptors using Keypoint detection  
            ▪ Employ Vectors of Locally Aggregated Descriptors as encoder  
            ▪ Train SVMs with both linear and RBF kernel  
            ▪ KAZE/VGG features extractor trained with linear SVM is the best performing SVM pipeline  
        ◦ CNN – trained with labelled images (higher accuracy with more computation complexity)  
            ▪ VGG-19 network architecture, replacing two linear layers by a Global Average Pooling (GAP) and two linear layers with dimension 4096 and 2048.  
            ▪ Data augmentation is also used  
            ▪ CNN has better accuracy (88.42%) compared to SVM pipeline  

4. Deep learning based automatic defect identification of photovoltaic module using electroluminescence images  
    • Using CNN-based deep learning model for defect classification with combination of traditional and generative deep-learning-based model (GAN) for data augmentations  
    • Traditional image augmentation used to produce produces new data with low computation time and hardware demand with more types of augmentation such as rotation, translation, and scaling features    
    • GAN-based image generation introduces more variation in new data  
    • The combination of two image augmentation method improves the classification accuracy significantly  
    • The proposed CNN-based model provides the best trade-off between accuracy and computational complexity.  
    • Noted that the model is created with optimal depth (increasing the depth decrease the accuracy) with optimal number of convolution kernel as well as inclusion of pooling layers that is able to reduce the computation complexity  

5. Solar Cell Surface Defect Inspection Based on Multispectral Convolutional Neural Network
    • Construct a multi-spectral CNN model to enhance the discrimination ability, in order to distinguish between complex texture background features and defect features. Accuracy of defect recognition reaches 94.3%.
    • The first paper to inspect solar cell defect using deep learning.
    • Methods:
    	• Multispectral defect feature analysis.
    	• CNN model design: Based on Alexnet model, adjusting the convolutional kernel size and network depth to increase the model's ability to defect discrimination. The deeper the model is, the better the features it can extract. The larger the kernel is, the more surrounding information it can extract from the features.
    	• Multi-spectral CNN model design: The 3 spectra are split are sent to 3 different networks, and their output characteristics are connected and put into a fully connected layer. The multi-spectral model can extract mixed features of multiple spectra, giving distinct features of each defect in the different spectra, so the outputs are easier to inspect defect.
    • Experiments:
   	• Select CNN depth, kernel size and stride step: Depending on the defect datasets, 3 different models with different depths and kernel sizes are designed and then select the best one. Evaluate using Precision, Recall and F-measure. The best structure gave 87.3% precision, 97.04% recall and 0.9187 F-measure. The step size is then selected as 469x469.
   	• Compare between multi-spectral model and normal model: Use K-fold cross validaton (K=5) to increase credibility of training results. 
   		• Result: Multi-spectral CNN model has higher detection rates of cell defects. Some defects results are about 1% higher. Different train-test ratio are also conducted, and the model is still effective with different split. Higher train-test ratio means better precision, recall and F-measure.
   	• Compare in multi-class classification: Multi-spectral CNN model has 2-6% higher accuracy compared to normal CNN model. The result of multi-class classification is 8% lower than binary classification.
   	• Compare with other ML methods: Compare with LBP+HOP-SVM and Gabor-SVM. MS-CNN has best results: 88.41% precision, 98.4% recall and 0.94 F-measure compared to other methods. Training and detection time is also experiments, and MS-CNN gave much better detection time than the other 2 methods, but with higher training time.

6.

7. Automated Detection of Solar Cell Defects with Deep Learning
    • Propose a deep CNN for EL cell image classification. Present a signal processing pipeline for image preprocessing and classification and different methods to handle dataset imbalance.
    • Methods:
    	• Pipeline: Correct the image distortions -> Segmentation -> Perspective correction (RANSAC) -> Cells are extracted and classified by CNN.	
    	• CNN architecture: VGG16 adaptation, with reduced number of filters and fully connected layer size, changed output layer, extra batch norm, ELU activation, L2-norm.
    • Dealing with dataset imbalance: a non-heuristic resampling method. Data augmentation is also used to reduce overfitting (random horizontal/vertical image flipping). An extra augmentation step is done to deal with rotation, translation and shearing.
    • Dataset: Split train-test by 90-10 ratio, highly imbalance as only 3.4% are defect.
    • Experiments:
    	• Oversampling and no data augmentation: Minority class is enlarged, good-defect ratio is 2:1.
    	• No oversampling and data augmentation: Diversity is enlarged, but still imbalanced.
    	• Oversampling and data augmentation: A more balanced and diverse dataset.
    • Results:
    	• With only minor oversampling and data augmentation: FNR is high (50.26% and 38.89%).
    	• Only oversampling: Performed well on training set with low BER, but poorly on validation set -> overfitted.	
    	• Only data augmentation: Reduced FPR and FNR, but High BER (19.57%).
    	• Combine both: Overfitting is reduced, low BER (7.73%), low FNR (12.96%), tolerable slightly increase in FPR.

8. Defect detection in atomic-resolution images via unsupervised learning with translational invariance
    • Develop an approach for defect detection and classification based on one-class SVM.
    • Introduce 2 schemes of image segmentation and data preprocessing.
    	• Crop the neighborhood of the unique maximum/minimum intensity site in the projection of the unit cell/repeat unit.
    	• Crop the Fourier filtered images and each cropped image is centered at the position of an atom column.
    	• Patterson function used as a feature extraction descriptor to greatly reduce the principle components, and thus, variability in input space due to segmentation errors.

9. Unsupervised Anomaly Detection for X-Ray Images
    • Use unsupervised methods to detect anomalies to evaluate X-ray images of hands. 
    • Propose a powerful preprocessing pipeline to reduce the effect of noise. 
    • Evaluate on multiple approaches and can also achieve satisfying results without labels.
    • Preprocessing: Cropping using Otsu Binarization -> Hand Localization by manually labelling bounding boxes and then fine-tuning a SSD with MobileNet -> Segmentation using Photoshop to achieve pixel-wise mask -> Image resizing (for GAN-based models) -> Data augmentation (horizontal/vertical flipping, channel-wise multiplication, rotation, scaling) -> Pad images to 512x512 (AE + DCGAN) or 128x128 (BiGAN + αGAN) pixels.
    • Models:
    	• AEs: Using a variant of reconstruction loss - masked reconstruction loss. 
    		• CAE: Fully CNNs, with Batch Norm and ReLU activation, spatial resolution 16x16, 512 channels.
    		• VAE: Additional loss term introduced as KLD between N (µ(x), Σ(x)) and N (0, I).
    		• Usage: Pixel-wise heatmap, expecting anomalies with high errors in highlighting regions hardest to reconstruct, obtain an image-wise score towards anomalous regions.
    	• GANs:	
    		• DCGAN: Fully connected CNNs.
    		• BiGAN/ALIGAN: Extends DCGAN with an extra encoder to encode real images to latent space.
    		• α-GAN: 4 sub-networks: Encoder, Code-discriminator, Generator, Decoder.
    		• Output of discriminator as anomaly score (mean over code discriminator and discriminator probability for α-GAN).
    • Experiments: 
    	• Single channel images, trained on images without anomaly, images split by patient.
    	• Train models from scratch without transfer learning, perform manual hyperparameter search on validation set to choose best performers per type w/r/t ROC-AUC.
    • Results:
    	• All models have best runs when data is fully preprocessed.
    	• Histogram equalization improves results of AE-based models consistently.
    	• Top-k (k = 200) loss values across all pixels does not improve the result.
    	• Best ROC-AUC score with 60.7% for α-GAN using the discriminator probability, and 57% for CAE with pixel-level anomaly scores for better intepretability.
    	• All methods with reconstruction loss were able to generate heatmaps of anomalous regions that non-experts can interpret.

10. DFR: Deep Feature Reconstruction for Unsupervised Anomaly Segmentation
    • Propose an unsupervised anomaly segmentation approach that can detect and segments out the anomalies in small and confined regions of images.
    	• Develop a multi-scale regional feature generator to generate multiple spatial context-aware representations for every subregion.
    	• Design a deep CAE to detect anomalies via fast feature reconstruction.
    • Pipeline: 
    	• Hierarchical image feature extraction: Use pre-trained CNN
    	• Multi-scale regional feature generation: Design a regional feature extractor to generate discriminative multi-scale representations for every subregion of the image.
    	• Deep feature reconstruction: Design a CAE to compress the multi-scale representation into a low-dimension latent space and reconstruct again.
    	• Scoring and segmentation: Detect all possible anomalous regions based on reproduced regional feature map and its ground truth.
    • Experiment:
    	• Comparison: CAE-based approaches, GAN-based approach, VAE-grad, CNN-FD, ST
    	• Architecture: 
    		• VGG19 pretrained on ImageNet.
    		• Regional feature generator: Align hierarchical CNN feature maps by nearest-neighbor interpolation, a mean filter with spatial size 4x4, stride 4.
    		• CAE.
    	• Evaluation: ROC-AUC, PRO-AUC

11. Detecting Anomalous Structures by Convolutional Sparse Models
    • Exploit convolutional sparse models to learn a dictionary of filters capturing structures of normal images, in order to check for anomalous regions.
    • Highlight that local group sparsity of coefficient maps is essential in anomaly detection.
    • Method:
    	• Preprocess: Treat low and high-frequency components separately.
    	• Compute convolutional sparse representation of high frequency components w.r.t a dictionary of filters.
    	• Compute indicator vector by concatenation, and anomalies are patches yielding outliers in these vectors.	
    • Experiment:
     	• Monitor both low and high frequency components.
    	• Exclusively monitor high frequency components. 