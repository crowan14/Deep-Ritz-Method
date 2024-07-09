import torch
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
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
  
        #define layers
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        self.output = nn.Linear( n , 1 , bias=False )
        
        self.act = nn.Tanh()
      
    def forward( self , x ):
        
        #two hidden layer network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
        y = self.output(y)
        
        x1 = torch.reshape( x[:,0] , (len(x),1) )
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        
        #enforce homogeneous Dirichlet boundary condition
        bc = torch.mul( torch.sin( np.pi * x1 ) , torch.sin( np.pi * x2 ) )
        
        y = torch.mul( bc , y )
        
        return y
  
    def gradient( self , x ):
        
        #compute solution and spatial gradient
        x.requires_grad = True
        u = self.forward(x)
        grad_u = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        return [ u , grad_u ]
    
    def energy( self , x ):
        
        stuff = self.gradient(x)
        u = stuff[0]
        grad_u = stuff[1]
        
        #two gradient components
        u_x = torch.reshape( grad_u[:,0] , (len(u),1) )
        u_y = torch.reshape( grad_u[:,1] , (len(u),1) )
        
        x1 = torch.reshape( x[:,0] , (len(x),1) )
        x2 = torch.reshape( x[:,1] , (len(x),1) )
        
        #energy functional (constant source term)
        Pi = torch.mean( 0.5 * ( torch.square(u_x) + torch.square(u_y) ) - u )
        
        return Pi
        
    
######################################################

#integration grid
pts = 50
x = np.linspace(0,1,pts)
X , Y = np.meshgrid( x , x )
col1 = np.reshape( X , (pts**2,) ) 
col2 = np.reshape( Y , (pts**2,) ) 
int_grid = np.zeros((pts**2,2))
int_grid[:,0] = col1
int_grid[:,1] = col2

int_grid = torch.tensor( int_grid , dtype=torch.float32 )

#network
u = Network()

#training parameters
epochs = 15000
lr = 1e-4
optimizer = torch.optim.Adam( u.parameters() , lr=lr )
losses = np.zeros(epochs)

#training loop
for i in range(epochs):
    
    optimizer.zero_grad()
    loss = u.energy( int_grid )
    loss.backward()
    optimizer.step()
    losses[i] = loss.item()
    
    if i % 500 == 0:
        print(f'Epoch {i}, Loss {losses[i]}')

plt.figure()
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('Energy')
plt.title('Training')
plt.show()


#evaluate solution at integration points
sol = u.forward(int_grid).detach().numpy()
Z = np.reshape( sol , (pts,pts) )

# Creating figure
fig = plt.figure(figsize = (14, 9))
ax = plt.axes(projection ='3d')
ax.plot_surface( X , Y , Z )
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
plt.title('Converged solution')
plt.show()






