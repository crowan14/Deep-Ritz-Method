import torch
import numpy as np
from torch import nn
import torch.optim as optim
import pandas as pd
import random
import matplotlib.pyplot as plt

plt.close('all')

######################################################

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        
        #hidden layer width
        n = 25
  
        #define operations used in forming the network
        self.layer_1 = nn.Linear( 1 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer feed forward neural network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        y = self.output(y)
        
        #enforce Dirichlet boundary on the two boundaries at x=0 and x=1
        y = torch.sin( np.pi * x ) * y
        
        return y
  
    #evaluate network and its spatial gradient at specified integration points
    def gradient( self , x ):
        
        x.requires_grad = True
        u = self.forward(x)
        u_x = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0];
        #u_xx = torch.autograd.grad( u_x , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0];
        
        return [ u , u_x ]
    
    #variational energy corresponding to 1D heat conduction with constant source tf(x)
    def energy( self , x ):
        
        #body force
        def f(y):
            vals = np.sin(3*np.pi*y)
            return vals
        
        stuff = self.gradient(x)
        u = stuff[0]
        u_x = stuff[1]
        
        #energy functional (integrated with mean operation)
        Pi = torch.mean( 0.5 * torch.square(u_x) + torch.mul( u , f(x.detach()) ) )
        
        return Pi
        
    
######################################################

#integration grid
arr = np.linspace(0,1,50)

#neural network input has to be column vector
x = torch.tensor( np.atleast_2d(arr).T , dtype=torch.float32 )

#initialize network
u = Network()

#training parameters
epochs = 15000
lr = 1e-4
optimizer = torch.optim.Adam( u.parameters() , lr=lr )
losses = np.zeros(epochs)

#training loop
for i in range(epochs):
    
    optimizer.zero_grad()
    loss = u.energy(x)
    loss.backward()
    optimizer.step()
    losses[i] = loss.item()
    
    if i % 500 == 0:
        print(f'Epoch {i}, Loss {losses[i]}')


plt.figure()
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Energy')
plt.title('Training')
plt.show()


#evaluate solution at integration points
sol = u.forward(x).detach().numpy()


plt.figure()
plt.plot(x.detach().numpy(),sol)
plt.xlabel('x')
plt.ylabel('Temperature/Displacement')
plt.title('Converged solution')
plt.show()













