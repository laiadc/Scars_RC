import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.sparse import random
from scipy.sparse.linalg import eigs
from sklearn.linear_model import LinearRegression,Ridge
from sklearn.metrics import mean_squared_error
import scipy.sparse
import pickle
import plotly
from plotly.graph_objs import graph_objs as go
from plotly.subplots import make_subplots
import plotly.offline as pyo
from tqdm import tqdm

pyo.init_notebook_mode()

np.random.seed(0)

class ComplexRidge:
    def __init__(self, alpha=0.0001):
        self.alpha = alpha
        
    def fit(self, X, Y):
        # Add bias (column of 1ns)
        X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        X_star = X.conj().T

        inv = scipy.linalg.inv(X_star.dot(X) + self.alpha*np.identity(X.shape[1]))
        self.beta = inv.dot(X_star).dot(Y)
        
    def predict(self, X):
        X = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
        pred = self.beta.T.dot(X.T).T

        return pred
    
    def mse(self, y_true, y_pred):
        mse = np.real(np.mean(np.conjugate(y_true - y_pred)\
                              *(y_true - y_pred)))
        return mse
    
class ComplexReservoirComputing():
    '''
    Class to define a Reservoir Computing Network. It also contains the functions to calculate the echo states,
    train the network and test it with unseen data.

    Attributes:
        n_min (int): Number of time steps to dismiss
        t_step (int): Number of time steps to let the network evolve
        W_in (sciypy.sparse matrix): (size (N,K)) Input connections
        W_back (scipy.sparse matrix): (size (N,L)) Back connections
        W (scipy.sparse.matrix): (size (N,N)) Reservoir matrix
        L (int): output size
        K (int): input size
        N (int): Number of nodes of the reservoir
        u (np.array): (size t_step,K)) input for training
        y_teach (np.array): (size (t_step,L)) output for training
        initial state (np.array): (size N) initial value of the internal states x(0)
        f (function): Activation function
        f_out (function): Activation function for the output
        states (np.array): Internal states
        predictor (np.array): Learning algorithm
    '''
    def __init__(self, n_min = None, t_step=None, W_in = None, W = None, W_back = None,
                 u = None, y_teach = None, initial_state = None, f = None, f_out = None,
                f_out_inverse =lambda x:x, restore = False, folder='trained_model'):
        
        # Training parameters
        self.n_min = n_min #dismissed time steps (int)
        self.t_step = t_step
        
        if restore:
            self.load(folder)
            
        else:
            # Matrices
            self.W_in = W_in #input connections 
            self.W_back = W_back
            self.W = W #adjacency matrix (matrix of size self.N x self.N)
            
            # Input and output
            self.u = u #input
            self.y_teach = y_teach #desired output of the network

            self.nu = None  # Noise given to the reservoir
            
            self.states = None # matrix containing the internal states of the reservoir:
                            # columns -> nodes; rows -> time steps
                
            self.predictor = None #Linear regressor predictor

        # Dimension values
        self.N = self.W.shape[0] #dimension of the reservoir, i.e, number of nodes (int)
        if self.W_back is not None:
            self.L = self.W_back.shape[1] #dimension of the output (int)
        else:
            self.L = 0
        if self.W_in is not None:
            self.K = self.W_in.shape[1] #dimension of the input (integer) (may be None)
        else:
            self.K = 0
        

        
        self.initial_state = initial_state #initial state of the reservoir
        self.f = f #activation function of the reservoir
        self.f_out = f_out #activation function of the output
        self.f_out_inverse = f_out_inverse # Inverse of the output activation function
    
    
    def evolve_trajectories(self, noise=False, boundary_noise=0.00001,
                            long_formula=False, C=0.49, a=0.9):
        '''
        Function to update the internal states x(n) with time. 
        Attributes:
            noise (boolean): If true, random noise is added to the output
            boundary_noise (float): Upper and lower bounds for the noise
            long_formula (boolean): If true, the long-memory update for the internal states is used
            C (float): Value of C for the long formula
            a (float): Value of a for the long formula
            
        '''
        # Set initial state
        self.states = self.initial_state
        x_prev = self.initial_state
        
        # If noise==True we add random uniform noise to the output
        if noise and self.y_teach is not None:
            nu = np.random.uniform(low=-boundary_noise, high=boundary_noise,
             size=self.y_teach.shape)
            y = self.y_teach + nu
            self.nu = nu
        else:
            y = self.y_teach
        
        # We update the internal states at each time step
        for n in tqdm(np.arange(self.t_step)):               
            if self.u is None: # If there is no input
                x = self.f(self.W.dot(x_prev) + self.W_back.dot(y[n-1]))
            elif self.y_teach is None: # If there is input but no output
                x = self.f(self.W.dot(x_prev) + self.W_in.dot(self.u[n]))
            else: # If there is input and output
                x = self.f(self.W.dot(x_prev) + self.W_in.dot(self.u[n]) + self.W_back.dot(y[n-1]))
            if long_formula: # We can also use the long-memory update
                x = (1.-C*a)*x_prev + C*x
            
            # We stack the new internal state
            self.states = np.vstack((self.states,x))
            x_prev = x

        self.states = self.states[1:,:]
    

    def dismiss_initial_states(self):
        '''
        Function to dismiss the n_min initial states
        '''
        self.states = self.states[self.n_min:,:]

    def dismiss_initial_output(self):
        '''
        Function to dismiss the n_min initial outputs
        '''
        self.y_teach = self.y_teach[self.n_min:]
    
    def augmented_x(self, M):
        '''
        Function to crate the vector \tilde{x} = (x, x^2)
        Args:
            M (np.array): Matrix containing the internal states
        Returns:
            (np.array): Augmented internal states
        '''
   
        aux1 = M[:,:int(self.N/2)]
        aux2 = M[:,int(self.N/2):]
    
        return np.hstack((aux1, aux2**2))
    
    
    def train(self, noise=False, boundary_noise=0.00001, ridge=True, alpha = 1e-12, 
              long_formula = False, C=0.49, a=0.9, augmented = False):
        '''
        Function to train the reservoir.
        Args:
            noise (boolean): If true, random noise is added to the output
            ridge (boolean): If ture, a Ridge regression is used instead of a linear regression
            alpha (float): alpha value for the Ridge regression
            long_formula (boolean): If true, the long-memory update for the internal states is used
            C (float): Value of C for the long formula
            a (float): Value of a for the long formula
            augmented (boolean): If true, the augmented x is used in the fit method
        Returns:
            MSE_vector (np.array): size L. Vector containing the mean square error for each dimension of the ouput
        '''
        
        # 1. Evolve internal states
        self.evolve_trajectories(noise=noise, boundary_noise=boundary_noise, 
                                long_formula = long_formula, C=C, a=a)
        # 2. Dismiss initial states and output
        self.dismiss_initial_states()
        self.dismiss_initial_output()
        # 3. Find W^out
        if ridge: # We can use a Ridge regression instead than a lenar rergression
            lm =  ComplexRidge(alpha=alpha)
        else: # Using a linear regression
            lm = ComplexRidge(alpha=0)
        if augmented: # Using the augmented states (x,x^2)
            lm.fit(self.augmented_x(self.states), self.f_out_inverse(self.y_teach)) # Fit method
            y = self.f_out(lm.predict(self.augmented_x(self.states))) # Predict output
        else:
            lm.fit(self.states, self.f_out_inverse(self.y_teach))  # Fit method
            y = self.f_out(lm.predict(self.states)) # Predict output
            
        #Compute MSE
        MSE_vector = 1/(self.t_step-self.n_min)* \
                    sum(np.conjugate(self.y_teach - y)*(self.y_teach - y))
        
        self.predictor=lm
        
        return MSE_vector

    
    def test(self, t_autonom=50, t_dismiss=100, u=None, use_W_back=True, y_true=None,
             long_formula=False, C=0.49, a=0.9, augmented = False, update_x = False):
        '''
        Function to test the reservoir with unseen data.
        Args:
            t_autonom (int): Number of time steps used to predict new data
            t_dismiss (int): Dissmissal time steps
            u (np.array): Input for future data
            use_W_back (boolean): If true, we use backwards connections
            y_true (np.array): True new data, used to calculate test MSE
            long_formula (boolean): If true, the long-memory update for the internal states is used
            C (float): Value of C for the long formula
            a (float): Value of a for the long formula
            augmented (boolean): If true, the augmented x is used in the fit method
       
        Returns:
            y_tot (np.array): predicted y for each time step
            mse (np.array): test mse
        '''
        # Set initial states and outputs to the time t_step-t_dismiss
        x_prev = self.states[-(t_dismiss-1)]
        y =self.y_teach[-(t_dismiss)]
        y_tot=y
        mse=None
        
        # Evolve the internal state for each time
        for n in tqdm(np.arange(t_dismiss + t_autonom)):
            if u is None: # If there is no input
                x = self.f(self.W.dot(x_prev) + self.W_back.dot(y))
            elif use_W_back: # If there is input and output
                x = self.f(self.W.dot(x_prev) + self.W_in.dot(u[n]) + self.W_back.dot(y))
            else: # If there is only input
                x = self.f(self.W.dot(x_prev) + self.W_in.dot(u[n]))
            if long_formula: # If we want to use the long-memory formula
                x = (1.-C*a)*x_prev + C*x
                
            x_prev = x
            if augmented: # If we want to use the augmented internal states (x,x^2)
                y = self.f_out(self.predictor.predict(self.augmented_x(x.reshape(1,-1)).reshape(1,-1))[0])
            else:
                y = self.f_out(self.predictor.predict(x.reshape(1,-1))[0]) # Predict output
            y_tot = np.vstack([y_tot,y])
            
            if update_x:
                if n>=t_dismiss: # Update the internal states
                    self.states = np.vstack((self.states,x))
                else: # Add the new internal states
                    self.states[-(t_dismiss-1) + n] = x
        
        y_tot = y_tot[ 1:,:]
        if y_true is not None: # If we have the true output, we calculate the test MSE
            mse= np.mean(np.conjugate(y_true-y_tot[t_dismiss:])*(y_true-y_tot[t_dismiss:]), axis=0)
        return y_tot,mse

    
    def save(self, folder):
        with open(folder + '/states.npy', 'wb') as f:
            np.save(f, self.states)
            
        with open(folder +'/y_teach.npy', 'wb') as f:
            np.save(f, self.y_teach)
        
        with open(folder +'/nu.npy', 'wb') as f:
            np.save(f, self.nu)
            
        with open(folder +'/u.npy', 'wb') as f:
            np.save(f, self.u)
            
        scipy.sparse.save_npz(folder +'/W.npz', self.W)
        scipy.sparse.save_npz(folder +'/W_in.npz', self.W_in)
        scipy.sparse.save_npz(folder +'/W_back.npz', self.W_back)
        pickle.dump(self.predictor, open(folder +'/W_out.sav', 'wb'))

    
    def load(self, folder):
        with open(folder +'/states.npy', 'rb') as f:
            self.states = np.load(f,allow_pickle=True)
        with open(folder +'/y_teach.npy', 'rb') as f:
            self.y_teach = np.load(f,allow_pickle=True)
        with open(folder +'/nu.npy', 'rb') as f:
            self.nu = np.load(f,allow_pickle=True)
        with open(folder +'/u.npy', 'rb') as f:
            self.u = np.load(f,allow_pickle=True)
            
        self.W = scipy.sparse.load_npz(folder +'/W.npz')
        self.W_in = scipy.sparse.load_npz(folder +'/W_in.npz')
        self.W_back = scipy.sparse.load_npz(folder +'/W_back.npz')

        self.predictor = pickle.load(open(folder +'/W_out.sav', 'rb'))

