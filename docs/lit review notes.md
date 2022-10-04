1. Exploratory Data Analysis of Image Texture and Statistical Features on Myocardium and Infarction Areas in Cardiac Magnetic Resonance Images  
    • Extracting CV-related features (mean pixel value, etc), statistical features (means, medians, sd) from areas of interest   
    • Extract “textual” features (texture of the region of interest) to measure properties such as smoothness, coarseness, and regularity. Described in three principal approaches: statistical (calculated by GLCM matrix), structural and spectral.  
    • In order to select features best describe and classify the data into two desired class, a Bayes classifier is trained, and Leave-one-out resampling is used. Bayesian is train with density function p(x|wi) and prior P(wi).  
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
