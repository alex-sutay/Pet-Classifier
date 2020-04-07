# Pet-Classifier
The scripts I used to create a neural network that identifies which of my pets is in a photo

The python script looks at a directory of photos and creates a matrix with each row representing a single photo. It also compresses the photos. 
The directory it creates them in is decided by a user prompt.
It creates 4 matrices by default, saving each in a .mat file. It creates one for each of the RGB values of each pixel in addition to one for the grayscale version.
After creating each matrix, it will prompt the user for a file name to save it as. It puts the file in the same directory as the photos and adds ".mat" itself

The main octave script is "pets.m", the others I borrowed from a class and run the basic calculations, such as the cost function
The octave script loads in three files specified in the code.
It assumes that the data in each file is saved as "X".
It automatically takes the last 20% of each matrix to form the test set and uses the first 80% as the training set
