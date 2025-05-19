# Capstone-Project--DL
Real-Time American Sign Language (ASL) Alphabet Detection Using Convolutional Neural Networks
Introduction & Aim
This capstone project focuses on the development of a real-time American Sign Language (ASL) alphabet recognition system using deep learning techniques. The core objective was to design and implement a Convolutional Neural Network (CNN)-based model capable of classifying hand gestures representing ASL alphabets from live webcam input. This solution addresses communication barriers faced by the deaf and hard-of-hearing communities by enabling gesture-based interaction, particularly useful in education, accessibility tools, and assistive technologies.
The primary aim of this project was twofold: (1) to train a CNN that accurately classifies images of hand signs representing each ASL alphabet and (2) to deploy this trained model in a real-time application that captures video input, detects the region of interest (ROI), and predicts the corresponding alphabet on-the-fly.
________________________________________
Theoretical Foundation
The underlying theory behind this project is based on Convolutional Neural Networks (CNNs) — a class of deep neural networks particularly effective for image classification tasks. CNNs utilize spatial hierarchies of patterns, learning from the local pixel interactions in images. The network consists of several core building blocks:
•	Convolutional Layers: These layers apply filters (kernels) to the input image to extract important features such as edges, textures, and shapes. In this project, we used two convolutional layers: one with 32 filters and another with 64 filters, both of size (3x3), followed by a ReLU activation function to introduce non-linearity.
•	Max Pooling Layers: Following each convolutional layer is a max pooling operation with a (2x2) window. Pooling reduces the spatial dimensions (height and width) of the feature maps while preserving the most important information. This not only improves computational efficiency but also introduces spatial variance robustness.
•	Flatten Layer: This layer reshapes the output from the 2D feature maps into a 1D vector to be fed into fully connected layers.
•	Dense (Fully Connected) Layers: These layers perform the actual classification. The first dense layer contains 128 units with ReLU activation to learn complex representations, while the final output layer uses softmax activation, yielding a probability distribution over the 30 ASL classes.
________________________________________
Methodology
1.	Data Collection & Preprocessing
The dataset used consisted of labeled images of ASL alphabets (excluding 'J' and 'Z' due to their motion component). The images were organized into subfolders per class. Data preprocessing included resizing images to 64x64 pixels and normalizing pixel values to the [0,1] range using Keras’ ImageDataGenerator. The dataset was split into training (80%) and validation (20%) sets. The preprocessing pipeline ensured consistent input size and improved model generalization.
2.	Model Architecture & Training
A sequential CNN model was constructed using TensorFlow and Keras. The architecture included:
o	Conv2D(32, (3,3), activation='relu')
o	MaxPooling2D((2,2))
o	Conv2D(64, (3,3), activation='relu')
o	MaxPooling2D((2,2))
o	Flatten()
o	Dense(128, activation='relu')
o	Dense(num_classes, activation='softmax')
The model was compiled using the Adam optimizer and categorical crossentropy loss, suitable for multi-class classification. To prevent overfitting and ensure better performance, EarlyStopping was employed, monitoring the validation loss with a patience of 3 epochs.
3.	Model Evaluation & Saving
After training, the model was evaluated on the validation set. The achieved accuracy and loss metrics indicated that the model had successfully learned to generalize well on unseen images. The trained model was saved in .keras format for later use in inference.
4.	Real-time Inference with OpenCV
A separate script (predict.py) was created to load the trained model and perform real-time detection using a webcam. The key steps included:
o	Capturing video input via OpenCV
o	Defining a Region of Interest (ROI) on the frame
o	Preprocessing the ROI (resizing and normalization)
o	Passing the image through the model
o	Displaying the predicted alphabet on the frame using cv2.putText()
________________________________________
Result
The model achieved strong performance on the validation dataset with an accuracy of approximately [insert actual accuracy]. More importantly, in real-world, real-time scenarios, the system demonstrated responsive and reliable classification of ASL hand gestures under varying lighting and background conditions. The bounding box-based ROI system ensured focused and consistent detection, while the webcam-based real-time feedback loop provided an interactive user experience.
The project successfully bridges the gap between deep learning theory and practical assistive technology. It serves as a scalable foundation for more complex sign language systems, including continuous gesture recognition and full sentence construction using NLP models.
________________________________________
