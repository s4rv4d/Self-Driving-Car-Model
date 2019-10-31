#importing libraries
import numpy as np
import random

#will help us to save the state of nn and load stored state of nn
import os

import torch
import torch.nn as nn
import torch.nn.functional as f #for loss functions, or activation
import torch.optim as optim #for optimiser for stoc gd
import torch.autograd as autograd # to convert variable containing tensors into variables containing tensor and gradient
from torch.autograd import Variable

#creating the architecture of the neural network
class Network(nn.Module):
     '''
          input_size for number of inputs, and nb_action for number of outputs
     '''

     def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        
        #here we create a new object and define its properties like input neurons and output
        self.input_size = input_size #5
        self.nb_action = nb_action #3

        #now to make full connections in our case since we are only using one hidden layer we have two full connections 
        # one between input layer and hidden layer and one between output layer and hidden layer
        self.fc1 = nn.Linear(input_size, 30) # input neurons connect to hidden layer neurons
        self.fc2 = nn.Linear(30,nb_action) #hidden layer neuron to output layer neurons


     #now to make the function for forward propogation
     def forward(self, state):
        #this will return q values for each possible action depending on inputs
        #we will use rectifier func
        x = f.relu(self.fc1(state)) #this gives us activated hidden neurons
        q_values = self.fc2(x) #we get q values or z values


        return q_values # for each action


#implementing experience replay diff from markhov decision process
class ReplayMemory(object):
    '''
          capacity is capacity of the memory to hold no.of transition
    '''

    def __init__(self, capacity):
        self.capacity = capacity #max no.of transitions
        self.memory = []

    
    def push(self,event):
        self.memory.append(event)

        #memory len should be less than capacity
        if len(self.memory) > self.capacity:
            #delete the oldest event stored
            del self.memory[0]

    
    def sample(self, batch_size):
        #batch_size is the size of sample to be chosen randomly
        #zip creates a list with tuples, and sample table selects random values of fixed size
        #if list = ((1,2,4),(3,5,6)) then zip(*list) = ((1,3),(2,5),(4,6))
        #so in our case we have 4  batches one four states last_state, for new_state , for rewards and for action
        #each batch contains two vals one for tensor and one for gradients
        samples = zip(*random.sample(self.memory,batch_size))
        #now we need to convert the sample into a pytorch variable
        #0 mean concatinate in one dimension not two

        return map(lambda x: Variable(torch.cat(x,0)),samples)


#implementing deep q learning  
class Dqn():
    '''
          the init inputs input_size and nb_actions is to initilize the network class and the gamma
          parameter is for the discount factor mostly with a value of 0.9 and that can be altered
    '''

    
    def __init__(self, input_size, nb_action, gamma):

        self.gamma = gamma
        #now we create a list called the reward window where each position contains the mean of the 100 actions from the observations for the ai
        self.reward_window = []
        #then we create the model of nn
        self.model = Network(input_size,nb_action)
        #then we create the object for the memory that is the replay memory class
        self.memory = ReplayMemory(100000)
        #then we create an optimiser to basically perform sto gd to update weights,so we have to connect it to our model by self.model.parameter()
        # then we also specify a learning rate
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        #when we init a tensor we always need to add a fake dimension(3 pos, and the orientations) and its should be at first pos we use unsqueeze
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0 #this will give the action of rotation either [0, 20 or -20]
        self.last_reward = 0 # either 1 or 0


    #now to create a function that selects the action
    def select_action(self, state):
        #each state will have three q values depending on which an action will be chosen
        #diff between softmax and rmax is that in softmax you can go through other q values using probability
        #also but in rmax it goes through one q value only

        #for softmax we need the q value outputs as an input
        #but those vales are given because of input state which is a torch tensor
        # so we need to convert the state into torch variable and exclude
        #the gradient by using Variable(state, volatile = True)
        probs = f.softmax(Variable(self.model(Variable(state, volatile = True))) * 100)
        #7 is the temperature parammeter  which helps into choosing the action
        #the higher the number the more sure it is in choosing the action
        #the lower the temperature parameter the less sure it is in choosing action
        action = probs.multinomial(1) # multi.. picks random value

        return action.data[0,0] # we do this so that we dont select the fake batch


    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        #this will give the best action to play with each input_state
        #and we use unsqueeze to match the dimension between batch_state and batch_action cause action doesnt have fake dim
        #0 for unsqueeze is fake dim for state and 1 for action
        #and after we match dim we kill fake dim with a squeeze command with 1 as para so we can kill fake of action
        #and we cal next output fot target so that we can cal td
        next_outputs = self.model(batch_next_state).detach().max(1)[0]#we get max of q values of next state
        #we need to get all the max of all possoble action so we detach all values and compare for max
        #check handbook for target formula
        target = self.gamma*next_outputs + batch_reward
        td_loss = f.smooth_l1_loss(outputs, target)  #we use this loss function
        #we use this value to back propogate and update the weights, keep in mind we also need to reinitialize
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph = True)
        #then we update the weight by the following
        self.optimizer.step()


    def update(self, reward, new_signal):
        #we update whenever we get a new state
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        #then we update memory
        #longtensors can hold ints
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            #then we learn 
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        #then we update the last action
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)

        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        
        return action

    
    def score(self):
        #returns the mean of the vals in the reward window
        #(len(self.reward_window) + 1) we are doing this so that the denominator is never zero and doesnt crash the program
        return sum(self.reward_window)/(len(self.reward_window) + 1)

    
    def save(self):
        torch.save({'state_dict':self.model.state_dict(),
                    'optimizer':self.optimizer.state_dict(),},'last_brain.pth')

    
    def load(self):
        #first to check if the file exists
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint...")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Done loading!")
        else:
            print("no checkpoint found")








