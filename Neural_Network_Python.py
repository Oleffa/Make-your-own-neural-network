
# coding: utf-8

# In[14]:

#imports
# read multiple files 
import glob
import scipy.misc
import matplotlib.pyplot
import numpy
import scipy.special #for sigmoid function
import matplotlib.pyplot
#get_ipython().magic('matplotlib inline')
#matplotlib.inline


# In[26]:

#neural network class definition
class neuralNetwork:
    
    #init the network
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #set number of nodes in each layer
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        #learning rate
        self.lr = learningrate
        #link weight matrices, weight from input to hidden layer (wih)
        # wight from hidden to output layer (who)
        #initializing the weight matrices with a normal distributed value around 0
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        #sigmoid function to scale the output of nodes
        #apply the sigmoid function to the input x and return it
        self.activation_function = lambda x:scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)
        
        pass
    
    #train the network
    def train(self, input_list, targets_list):
        #conver input into a 2d array (T transforms matrix)
        inputs = numpy.array(input_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T
        
        #calculate signals into hidden layer
        hl_inputs = numpy.dot(self.wih, inputs)
        #signals from hidden layer
        hl_outputs = self.activation_function(hl_inputs)
        #signals into output layer
        ol_inputs = numpy.dot(self.who, hl_outputs)
        #signals from the output layer
        ol_outputs = self.activation_function(ol_inputs)
        
        #calculate errors (target - actual value)
        output_errors = targets - ol_outputs
        #hidden layer error is the output_errors split by weights, recombined at hidden nodes
        hidden_errors = numpy.dot(self.who.T, output_errors)
        
        #update weights for the links between the hidden and output layers
        self.who += self.lr*numpy.dot((output_errors*ol_outputs*(1.0-ol_outputs)), numpy.transpose(hl_outputs))
        #update weights for the links between the input and hidden layers
        self.wih += self.lr*numpy.dot((hidden_errors*hl_outputs*(1.0-hl_outputs)), numpy.transpose(inputs))
        
        pass
    
    #query the neutral network
    def querry(self, input_list):
        #convert the input in a 2d array (T transforms atrix)
        inputs = numpy.array(input_list, ndmin=2).T
        
        #signals into hidden layer
        hl_inputs = numpy.dot(self.wih, inputs)
        #signals from hidden layer
        hl_outputs = self.activation_function(hl_inputs)
        #signals into output layer
        ol_input = numpy.dot(self.who, hl_outputs)
        #signal from output layer
        ol_output = self.activation_function(ol_input)
        return ol_output
    # backquery the neural network
    #working like querry just backwards
    def backquery(self, targets_list):
        # transpose the targets
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale the values back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs


# In[27]:

#load the test data
test_data_file = open("/home/oli/Downloads/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close
#get the first test record
all_values = test_data_list[0].split(',')
#print the lable
#image_array = numpy.asfarray(all_values[1:]).reshape([28,28])
#matplotlib.pyplot.imshow(image_array, cmap='Greys', interpolation='None')


# In[28]:

#init and first test of neural network
input_nodes = 784
hidden_nodes =200
output_nodes = 10
learning_rate = 0.2
#create instance of the network
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
#load the training data file
training_data_file = open("/home/oli/Downloads/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()
print("hi")
#train the neural network
#epochs are the number of learning processes (the number of times we iterate through the whole set)
epochs = 5
for e in range(epochs):
#go through all datasets
    i = 1
    for record in training_data_list:
        #split input by comma to get the number and data corresponding to it
        all_values = record.split(',')
        #scale and shift inputs
        inputs = (numpy.asfarray(all_values[1:]) / 255.0*0.99) + 0.01
        #create the target output values
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        n.train(inputs, targets)
        print("epoch: ", e, " : ", i, " out of: 60.000")
        i = i + 1
        pass
    print(e)
    pass
print("done")


# In[17]:

#test the neural network and see how it scores
#save the scores, initially empty
scorecard = []
#go through all the records in the test dataset
for record in test_data_list:
    #split by ,
    all_values = record.split(',')
    #correct answer is first value
    correct_label = int(all_values[0])
    #scale and shift the inputs
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    #query 
    outputs = n.querry(inputs)
    #the index of the highest value corresponds to the label
    label = numpy.argmax(outputs)
    #print("solution: ", correct_label, "guess: ", label)
    #append correct or incorrect to list
    if(label == correct_label):
        #if the answer was correct add 1 to scorecarde
        scorecard.append(1)
        print("CORRECT: solution: ", correct_label, "guess: ", label)
    else:
        #if the answer is wrong add 0
        scorecard.append(0)
        print("WRONG: solution: ", correct_label, "guess: ", label)
    pass
pass
#calculate performance score
scorecard_array = numpy.asarray(scorecard)
print("performance = ", scorecard_array.sum() / scorecard_array.size)


# In[18]:

# load own handwriting
our_own_dataset = []

# load the png image data
for image_file_name in glob.glob('/home/oli/Make-your-own-neural-network/own_handwriting/?.png'):
    
    # use the filename to set the correct label
    label = int(image_file_name[-5:-4])
    
    # load image data from png files into an array
    print ("loading ... ", image_file_name)
    img_array = scipy.misc.imread(image_file_name, flatten=True)
    
    # reshape from 28x28 to list of 784 values, invert values
    img_data  = 255.0 - img_array.reshape(784)
    
    # then scale data to range from 0.01 to 1.0
    img_data = (img_data / 255.0 * 0.99) + 0.01
    print(numpy.min(img_data))
    print(numpy.max(img_data))
    
    # append label and image data  to test data set
    record = numpy.append(label,img_data)
    our_own_dataset.append(record)

    
    pass


# In[ ]:

# test the neural network with our own images
# record to test
item = 0

# plot image
matplotlib.pyplot.imshow(our_own_dataset[item][1:].reshape(28,28), cmap='Greys', interpolation='None')

# correct answer is first value
correct_label = our_own_dataset[item][0]
# data is remaining values
inputs = our_own_dataset[item][1:]

# query the network
outputs = n.querry(inputs)
print (outputs)

# the index of the highest value corresponds to the label
label = numpy.argmax(outputs)
print("network says ", label)
# append correct or incorrect to list
if (label == correct_label):
    print ("match!")
else:
    print ("no match!")
    pass


# In[32]:

#run the backquery and see its output

# test input into the output layer
label = 9
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99
print(targets)

# get image data
image_data = n.backquery(targets)

# plot image data
matplotlib.pyplot.imshow(image_data.reshape(28,28), cmap='Greys', interpolation='None')
print("done")


# In[ ]:



