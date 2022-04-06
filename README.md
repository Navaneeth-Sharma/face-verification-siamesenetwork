# Face Verification Siamese Network

Face Verification is a complex problem and is evolving every year. There are many challenges one can face while creating the Face Verifcation System. To solve this problem there are two popular approches.

1. Face Verification using Facenet
2. One Shot Approach (Siamese Network based Approach)

However for normal face verification system both work quit well. For Faces with mask, glasses there is an ambiguity b/w choosing the approach. Here I am using the **One Shot Learning approach** to solve this problem.

## Block Diagram

![](docs/items/Blank%20diagram.png)

## Architecture of Siamese Network

The siamese network is a network which has shared weights b/w all the data. There are multiple architectures to build this network. Trying with multiple networks the **Resnet34** based architecture gives the good result. The training accuracy is about 87% and validation accuracy(here the data is subset of the train data not test data) is about 80%.

![](docs/items/Untitled%20Diagram.drawio.png)

## Loss Function Used

The Triplet Loss function is used to achive the results.


## Steps to run

- Put the images in images folder
- run the download_models.sh which is in the models folder
- finally run the main.py
