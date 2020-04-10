# Pet-Classifier
The scripts I used to create a neural network that identifies which of my pets is in a photo

The python script pic_to_vec.py looks at a directory of photos and creates a matrix with each row representing a single photo. It also compresses the photos. 
The directory it creates them in is decided by a user prompt.
It creates 4 matrices by default, saving each in a .mat file. It creates one for each of the RGB values of each pixel in addition to one for the grayscale version.
After creating each matrix, it will prompt the user for a file name to save it as. It puts the file in the same directory as the photos and adds ".mat" itself
It imports numpy for matrix construction and manipulation, scipy.io for octave compatibility, Image from PIL for image manipulation, and os to open the files.
Of the above imports, you will need to pip install pillow and scipy

predictImage.py uses the vectorization methods in pic_to_vec.py and a set of matrices representing a neural network to come up with a prediction for an image
It has a method thetas_from_mat which should extract variables representing the neural network and put them in alphabetical order
Because the order of the variables is assumed to be alphabetical, it's suggested you use variables with the same name and a number at the end denoting the order. 
For example, a neural net with one hidden layer might save theta1 and theta2 as the parameters, while a network with 2 hidden layers might use theta1, theta2 and theta3.

classifierBot.py is a discord bot script that uses pic_to_vec.py and predictImage.py.
Whenever someone uploads a photo that the bot can see, it will download it and save it as "image".
Once it has the image, it will run it through predictImage.predict to get a numerical output, then check the KEY dictionary for the title of that output.
classifierBot.py requires a created config file, config.py, to work. Inside config.py, it needs 4 variables to be defined:
TOKEN: the bot token provided by discord
THETA_FILE: the name of the file containing the parameters for the neural network
CHANNEL: the color channel you want to use. It can be 'r', 'g', 'b', or 'l'(grayscale)
KEY: the answer key for what each number output represents. It should be a dictionary of integers to strings
In addition to the requirements above, running classifierBot.py will require you pip install discord

The main octave script is "pets.m", the others I borrowed from a class and run the basic calculations, such as the cost function
The octave script loads in three files specified in the code.
It assumes that the data in each file is saved as "X".
It automatically takes the last 20% of each matrix to form the test set and uses the first 80% as the training set
