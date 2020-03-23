#Raunaq Foridi 2019
#This program has been written to be importable as a module.
#If being used directly, this will act as debugging and testing.
import math
import numpy
import random
import copy
input_layer=[]
hidden_layer=[]
output_layer=[]
hidden_dummy=[]
output_dummy=[]
weights=[[],[]]
bias=[[],[]]
set_weights=0
set_bias=0
set_weights_check=False
set_bias_check=False
avg_cost=[]
training_output=0.5
learning_rate=0.01
training_input=0.5
delta=0.0000001
gradient_warning=0
range_multiplier=1
def sigmoid(x):
    #compresses any number to a number between 0 and 1.
    try:
        res = 1 / (1 + math.exp(-x))
    except OverflowError:
        res = 0.0
    return res
def inv_sigmoid(y):
    #undoes the sigmoid function-useful for debugging
    res = math.log(y/(1-y))
    return res    
def initialise_network(input_size,hidden_size,output_size):
    #creates the network, with variable amount of neurons in each layer
    global input_layer
    global hidden_layer
    global output_layer
    global weights
    global bias
    global output_dummy
    global hidden_dummy
    input_layer=[]
    hidden_layer=[]
    output_layer=[]
    weights=[[],[]]
    bias=[[],[]]
    i=0
    x=0
    #initialise the weights list - add a new nested list for each neuron
    #[i][j][k] = [Layer][Neuron][Weight]
    while i<input_size:
        input_layer.append(0)
        weights[0].append([])
        x=0
        while x<hidden_size:
            #assign each weight a random value
            weights[0][i].append(random.random()*range_multiplier)
            x+=1
        i+=1
    i=0
    x=0
    #repeat for the hidden layer
    while i<hidden_size:
        hidden_layer.append(0)
        hidden_dummy.append(0)
        weights[1].append([])
        bias[0].append(random.random())
        x=0
        while x<output_size:
            weights[1][i].append(random.random()*range_multiplier)
            x+=1
        i+=1
    i=0
    #initialise all outputs to 0
    while i<output_size:
        output_layer.append(0)
        output_dummy.append(0)
        bias[1].append(random.random()*range_multiplier)
        i+=1
def determine_activation(inputs):
    #determine activations given a certain input. takes a list as an input.
    global input_layer
    global hidden_layer
    global output_layer
    global weights
    global bias
    global output_dummy
    global hidden_dummy
    i=0
    output_layer=copy.deepcopy(output_dummy)
    hidden_layer=copy.deepcopy(hidden_dummy)
    if len(inputs)!=len(input_layer):
        print("Invalid input- network initialised with "+str(len(input_layer))+" inputs")
        return
    else:
        input_layer=inputs
    #calculate each neuron of the hidden layer
    while i<len(hidden_layer):
        x=0
        while x<len(weights[0]):
            hidden_layer[i]+=float(input_layer[x])*float(weights[0][x][i])
            x+=1
        hidden_layer[i]=sigmoid(hidden_layer[i]+bias[0][i])
        i+=1
    i=0
    #repeat for the output layer
    while i<len(output_layer):
        x=0
        while x<len(weights[1]):
            output_layer[i]=float(output_layer[i])+float(hidden_layer[x])*float(weights[1][x][i])
            x+=1
        output_layer[i]=sigmoid(output_layer[i]+bias[1][i])
        i+=1
    return output_layer

def cost(intended_output):
    #calculate the "cost" of the system - how incorrect it is.
    global output_layer
    if len(output_layer)!=len(intended_output):
        print("intended output is the wrong length! should match Output size of "+str(len(output_layer)))
    vector1=numpy.array(intended_output)
    vector2=numpy.array(output_layer)
    vector3=(vector1-vector2)**2
    return numpy.sum(vector3)

def gradient_nudge():
    #will nudge each weight, one by one, and see its effects
    #then, times the difference by a learning rate and subtract from original
    #This function has been deprecated. It is inefficient and buggy.
    global gradient_warning
    global weights
    global bias
    i=0
    x=0
    n=0
    if gradient_warning==0:
        print("This function has been deprecated. use 'backprop_nudge()' as it actually uses backpropogation")
        gradient_warning=1
    while i<2:
        x=0
        while x<len(weights[i]):
            n=0
            while n<len(weights[i][x]):
                determine_activation(training_input)
                old_cost=cost(training_output)
                weights[i][x][n]+=delta
                determine_activation(training_input)
                updated_cost=cost(training_output)
                gradient=(updated_cost-old_cost)/delta
                weights[i][x][n]-=delta
                weights[i][x][n]-=gradient*learning_rate
                determine_activation(training_input)
                new_cost=cost(training_output)
                if new_cost>old_cost:
                    weights[i][x][n]+=2*gradient*learning_rate
                if cost(training_output)>new_cost:
                    weights[i][x][n]-=gradient*learning_rate
                n+=1
            x+=1
        i+=1
    i=0
    x=0
    while i<2:
        x=0
        while x<len(bias[i]):
            old_cost=cost(training_output)
            bias[i][x]+=delta
            determine_activation(training_input)
            updated_cost=cost(training_output)
            gradient=(updated_cost-old_cost)/delta
            bias[i][x]-=delta
            bias[i][x]-=gradient*learning_rate
            determine_activation(training_input)
            new_cost=cost(training_output)
            if new_cost>old_cost:
                bias[i][x]+=2*gradient*learning_rate
            if cost(training_output)>new_cost:
                    bias[i][x]-=gradient*learning_rate
            x+=1
        i+=1
def backprop_nudge(save=0):
    #simulate backpropogation. The "save" parameter is useful for large datasets
    #and will be used to take averages, as opposed to a "per case" basis.
    global weights
    global bias
    global input_layer
    global hidden_layer
    global output_layer
    #first, find the derivative of the output against the cost function
    #then, find the derivative of the pre-sigmoid 
    #then, the derivative of the weights, biases and previous activation against the pre-sigmoid
    determine_activation(training_input)
    
    old_cost=cost(training_output)

    output_layer2=copy.deepcopy(output_layer)
    
    output_layer=[output_layer[x]+delta for x in output_layer2]
    
    derivative_output_cost=[delta/(cost(training_output)-old_cost)[x] for x in cost(training_output)]
    
    derivative_sigmoid_output=[delta/(sigmoid(inv_sigmoid(output_layer2[x])+delta)-output_layer2[x]) for x in output_layer2]
    
    derivative_weight_sigmoid=copy.deepcopy(weights)
    for i in range(len(weights[1])):
        for j in range(len(weights[1][i])):
            weights[1][i][j]+=delta
            derivative_weight_sigmoid[1][i][j]=determine_activation(training_input)[j]/output_layer2[j]
            derivative_weight_sigmoid[1][i][j]=delta/(determine_activation(training_input)[j]-output_layer2[j])
            weights[1][i][j]-=delta
    derivative_bias_sigmoid

        
if __name__=="__main__":
    while True:
        choice=input("type in desired mode: ").lower()
        if choice=="change delta":
            delta=float(input("Enter a value for Delta. Lower is more accurate."))
        elif "set weight" in choice or "sw" in choice:
            set_weights=eval(input("please enter the desired starting weights, as a list: "))
            set_weights_check=True
        elif "set bias" in choice or "sb" in choice:
            set_bias=eval(input("please enter the desired starting biases, as a list: "))    
            set_bias_check=True
        elif "range" in choice or choice=="rm":
            range_multiplier=float(input("Please enter a multiplier for the range of starting weights: "))
        elif choice=="1 neuron match test" or choice=="1nmt":
            initialise_network(1,1,1)
            if set_weights_check:
                weights=copy.deepcopy(set_weights)
                set_weights=0
            if set_bias_check:
                bias=copy.deepcopy(set_bias)
                set_bias=0
            reps=int(input("Network initialised with 1 input, 1 hidden neuron and 1 input. How many training repetitions would you like to perform? "))
            learning_rate=float(input("please enter the desired learning rate: "))
            print("Now training the Network. This Test Network will be trained to Output The double Sigmoid of Whatever was Inputted into it.")
            print("perfect for Debugging.")
            i=0
            while i<reps:
                training_input=[random.random()]
                training_output=[sigmoid(sigmoid(training_input[0]))]
                if i%5==0:
                    print("training on "+str(training_input)+",current cost is: "+str(cost([training_output]))+" rep #"+str(i+1))
                determine_activation(training_input)
                gradient_nudge()
                i+=1
                print("completed repetition "+str(i))
            print("completed training. Current cost is....")
            print(cost([training_output]))
            set_bias_check=False
            set_weights_check=False
        elif "xor" in choice or "perceptron" in choice:
            initialise_network(2,1,1)
            if set_weights_check:
                weights=copy.deepcopy(set_weights)
                set_weights=0
            if set_bias_check:
                bias=copy.deepcopy(set_bias)
                set_bias=0
            reps=int(input("Network initialised with 2 inputs, 1 hidden neuron and an output. Please enter how many repetitions to perform: "))
            learning_rate=float(input("this network will take 2 inputs and applied 'XOR' to them. The hidden layer will be taken as the output. Please enter the learning rate: "))
            print("Now training the Network. This Test Network will be trained to output '1' if the 2 inputs are different, and '0' otherwise.")
            i=0
            while i<reps:
                training_input=[random.random()]
                training_output=sigmoid(sigmoid(training_input))
                print("training on "+str(training_input)+",current cost is: "+str(cost([training_output]))+" rep #"+str(i+1))
                determine_activation([training_input])
                gradient_nudge()
                i+=1
                print("completed repetition "+str(i))
            set_bias_check=False
            set_weights_check=False
        elif ("0t" in choice or "zero test" in choice) and "x" not in choice:
            initialise_network(1,1,1)
            print(hidden_layer)
            if set_weights_check:
                weights=copy.deepcopy(set_weights)
                set_weights=0
            if set_bias_check:
                bias=copy.deepcopy(set_bias)
                set_bias=0
            reps=int(input("Network initialised with 1 input, 1 hidden neuron and 1 output. How many training repetitions would you like to perform? "))
            learning_rate=float(input("please enter the desired learning rate: "))
            print("Now training the Network. This Test Network will be trained to output Zero, every time.")
            print("perfect for Debugging.")
            for i in range(reps):
                training_input=[random.random()]
                training_output=[0.5]
                determine_activation(training_input)
                gradient_nudge()
                if i%100==0:
                    print("rep #"+str(i)+", testing on "+str(round(training_input[0],4))+", Current cost: "+str(cost([0.5])))
            set_bias_check=False
            set_weights_check=False
        elif "0tx" in choice or "zero test x" in choice:
            print("This mode Mirrors the Zero Test, but with a variable amount of output layers")
            initialise_network(1,1,int(input("Please input the number of Output Layers: ")))
            if set_weights_check:
                weights=copy.deepcopy(set_weights)
                set_weights=0
            if set_bias_check:
                bias=copy.deepcopy(set_bias)
                set_bias=0
            reps=int(input("Network initialised with 1 input, 1 hidden neuron and %s outputs. How many training repetitions would you like to perform? " % str(len(output_layer))))
            learning_rate=float(input("please enter the desired learning rate: "))
            training_output=[]
            for i in range(len(output_layer)):
                training_output.append(0.5)
            for i in range(reps):
                training_input=[random.random()]
                determine_activation(training_input)
                gradient_nudge()
                if i%100==0:
                    print("rep #"+str(i)+", testing on "+str(round(training_input[0],4))+", Current cost: "+str(cost(training_output)))
            set_bias_check=False
            set_weights_check=False
        else:
            print("mode name not recognised. Allowing Manual Input")
            break
