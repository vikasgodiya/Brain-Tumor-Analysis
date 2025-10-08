<h1 align="left">🧠 Brain Tumor MRI Classifier</h1>

###

<p align="left"></p>

###

<p align="left">Building a detection model using a convolutional neural network in Tensorflow & Keras.<br>Used a brain MRI images data founded on Kaggle. You can find it here.<br><br>About the data:<br>The dataset contains 2 folders: yes and no which contains 253 Brain MRI Images. The folder yes contains 155 Brain MRI Images that are tumorous and the folder no contains 98 Brain MRI Images that are non-tumorous.</p>

###

<h2 align="left">Getting Started</h2>

###

<h3 align="left">Data Augmentation:</h3>

###

<p align="left">Why did I use data augmentation?<br><br>Since this is a small dataset, There wasn't enough examples to train the neural network. Also, data augmentation was useful in taclking the data imbalance issue in the data.<br><br>Further explanations are found in the Data Augmentation notebook.<br><br>Before data augmentation, the dataset consisted of:<br>155 positive and 98 negative examples, resulting in 253 example images.<br><br>After data augmentation, now the dataset consists of:<br>1085 positive and 980 examples, resulting in 2065 example images.<br><br>Note: these 2065 examples contains also the 253 original images. They are found in folder named 'augmented data'.</p>

###

<p align="left"></p>

###

<h3 align="left">Data Preprocessing</h3>

###

<p align="left">For every image, the following preprocessing steps were applied:<br><br>Crop the part of the image that contains only the brain (which is the most important part of the image).<br>Resize the image to have a shape of (240, 240, 3)=(image_width, image_height, number of channels): because images in the dataset come in different sizes. So, all images should have the same shape to feed it as an input to the neural network.<br>Apply normalization: to scale pixel values to the range 0-1.</p>

###

<h3 align="left">Data Split:</h3>

###

<p align="left">The data was split in the following way:<br><br>70% of the data for training.<br>15% of the data for validation.<br>15% of the data for testing.</p>

###

<h3 align="left">Neural Network Architecture</h3>

###

<p align="left">Understanding the architecture:<br>Each input x (image) has a shape of (240, 240, 3) and is fed into the neural network. And, it goes through the following layers:<br><br>A Zero Padding layer with a pool size of (2, 2).<br>A convolutional layer with 32 filters, with a filter size of (7, 7) and a stride equal to 1.<br>A batch normalization layer to normalize pixel values to speed up computation.<br>A ReLU activation layer.<br>A Max Pooling layer with f=4 and s=4.<br>A Max Pooling layer with f=4 and s=4, same as before.<br>A flatten layer in order to flatten the 3-dimensional matrix into a one-dimensional vector.<br>A Dense (output unit) fully connected layer with one neuron with a sigmoid activation (since this is a binary classification task).<br>Why this architecture?<br><br>Firstly, I applied transfer learning using a ResNet50 and vgg-16, but these models were too complex to the data size and were overfitting. Of course, you may get good results applying transfer learning with these models using data augmentation. But, I'm using training on a computer with 6th generation Intel i7 CPU and 8 GB memory. So, I had to take into consideration computational complexity and memory limitations.<br><br>So why not try a simpler architecture and train it from scratch. And it worked :)</p>

###

<h3 align="left">Results</h3>

###

<p align="left">Now, the best model (the one with the best validation accuracy) detects brain tumor with:<br><br>88.7% accuracy on the test set.<br>0.88 f1 score on the test set.<br>These resutls are very good considering that the data is balanced.<br><br>Performance table of the best model:<br><br>Accuracy on Validation Set : 91%<br>Accuracy on Test Set : 89%<br>F1 Score on Validation Set  : 0.91<br>F1 Score on Test Set : 0.88</p>

###

<p align="left"></p>

###

<h3 align="left">Final Notes To Ponder</h3>

###

<p align="left">This files contains : <br><br>1. Coding is done in ipynb notebook<br>2. The weights for all the models. The best model is named as mobilenetv2_best_model.h5<br>3. A Streamlit webapp named bt.py can be run to access on web by using "streamlit run bt.py" in terminal</p>

###

<p align="left"></p>

###

<div>
  <img style="100%" src="https://capsule-render.vercel.app/api?type=waving&height=100&section=header&reversal=false&fontSize=70&fontColor=FFFFFF&fontAlign=50&fontAlignY=50&stroke=-&descSize=20&descAlign=50&descAlignY=50&theme=cobalt"  />
</div>

###
