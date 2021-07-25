%% Expt. 11 Image Classification using GoogLeNet CNN 

%{ 
THEORY:
GoogLeNet is a convolutional neural network that is 144 layers deep. 
It is trained on more than 14 million hand-annotated images 
You can load a pretrained version of this trained network. 
The network trained on ImageNet classifies images into 1000 object
categories, such as keyboard, mouse, pencil, and many animals. 
The pretrained networks has an image input size of 224x224x3.
GooLeNet has 6.7977 million parameters(weights)
%}

%% Load the pretrained CNN named google net
net = googlenet;

%% See details of the architecture
net.Layers

%% Read and Resize Image

% The image that you want to classify must have the same size as the input size of the network. 
% For GoogLeNet, the network input size is the InputSize property of the image input layer.
%I = imread("football.jpg");
 I = imread("autumn.tif")
inputSize = net.Layers(1).InputSize;
I = imresize(I,inputSize(1:2));

%% Extract all the class names and  displayed them
classNames = net.Layers(end).ClassNames;
disp(classNames) %display  class names

%% Classify and display the image with the predicted label.
[label,scores] = classify(net,I);
figure
imshow(I)
title(string(label) + ", " + num2str(100*scores(classNames == label),3) + "%");


%% displaying top 5 concepts identified with scores for the given image
[ss,ii]=sort(scores,'descend');
ss5=ss(1:5);
label5=classNames(ii(1:5));
disp('------------------------------------')
disp('Top five classes identifed by the googLeNet for the given image with confidence scores:')
for i = 1 : 5
    disp( string(classNames(ii(i))) + ":" + num2str(ss5(i)*100));
end





