import numpy as np
from scipy.integrate import solve_ivp
import csv

class IntegrateSchrodinger2D:
    '''
    Class to integrate the Schrodinger equation for arbitrary 1D potentials.
    We use the Kosloff method based on the FFT
    '''
    def __init__(self, x, y, psi0, V, k0 = None, hbar=1, m=1, t0=0.0, dt=0.01):
        
        # Validation of array inputs
        self.x, self.y, psi0, self.V = map(np.asarray, (x, y, psi0, V))
        self.Nx = self.x.shape[1]
        self.Ny = self.y.shape[0]
        assert self.x.shape == (self.Ny,self.Nx)
        assert self.y.shape == (self.Ny,self.Nx)
        assert psi0.shape == (self.Ny,self.Nx)
        assert self.V.shape == (self.Ny,self.Nx)
        
        # Set internal parameters
        self.hbar = hbar
        self.m = m
        self.t = t0
        self.dt = dt
        self.dx = self.x[0,1] - self.x[0,0]
        self.dy = self.y[1,0] - self.y[0,0]
        self.dkx = 2 * np.pi / (self.Nx * self.dx)
        self.dky = 2 * np.pi / (self.Ny * self.dy)
        
        # Set minimum momentum
        if k0 == None:
            self.k0x = -np.pi/self.dx
            self.k0y = -np.pi/self.dy
        else:
            self.k0x = k0
        # Set momentum grid
        self.kx = self.k0x + self.dkx * np.arange(self.Nx)
        self.ky = self.k0y + self.dky * np.arange(self.Ny)
        
        self.kx, self.ky = np.meshgrid(self.kx, self.ky)
        
        # Define psi_xy and psi_xy_old
        self.psi_xy = psi0
        self.psi_xy_old = psi0 # Store old value of psi_xy for finite difference
        self.psi_xy_all = psi0.reshape(1,self.Ny,self.Nx)
        self.dt = dt
        

    
    def compute_psi_k_from_psi_x(self, psi_xy):
        '''
        FFT to psi(x,t)
        We define psi(x,t) multiplied by the term so that psiF_x and psiF_k are Fourier pairs
        '''
        psiF_xy = (psi_xy * np.exp(-1j * self.kx[0,0] * self.x)* np.exp(-1j * self.ky[0,0] * self.y) \
                   *(self.dx *self.dy)/ (2 * np.pi))
        
        psiF_k = np.fft.fft2(psiF_xy)
    
        Nx,Ny = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
        
        psi_k = psiF_k * np.exp(-1j * self.x[0,0] * self.dkx * Nx)* \
                np.exp(-1j * self.y[0,0] * self.dky * Ny)
        
        return psi_k


    def compute_psi_x_from_psi_k(self, psi_k):
        '''
        Inverse FFT to psi(k,t)
        We define psi(x,t) multiplied by the term so that psiF_x and psiF_k are Fourier pairs
        '''
        Nx,Ny = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny))
        
        psiF_k = psi_k* np.exp(1j * self.x[0,0] * self.dkx * Nx)* \
                np.exp(1j * self.y[0] * self.dky * Ny)
        
        psiF_xy = np.fft.ifft2(psiF_k)
        
        psi_xy = (psiF_xy* np.exp(1j * self.kx[0,0] * self.x) * \
                  np.exp(1j * self.ky[0,0] * self.y)* (2 * np.pi) / (self.dx*self.dy))
        
        return psi_xy
    
    
    def diff_RK(self, t, psi_xy):
        '''
        Function to pass to the Runge-Kutta method
        '''
        # 1. Calculate nabla^2psi(x,t) by:
        # a) Obtain psi(k,t) using FFT
        psi_k = self.compute_psi_k_from_psi_x(psi_xy.reshape(self.Ny,self.Nx))

        # b) Multiply psi(k,t) by hbar^2 k^2/(2m)
        nabla_psi_k = -psi_k*(self.kx**2 + self.ky**2)

        # c) Obtain the kinetic part in the x space using the inverse FFT
        nabla_psi_xy = -self.hbar**2/(2*self.m)*self.compute_psi_x_from_psi_k(nabla_psi_k)

        # 2. Add V(x)psi(x,t) to obtain H*psi(x,t)
        H_psi = nabla_psi_xy +  self.V*self.psi_xy
        
        return (-1j/self.hbar*H_psi).flatten()

        
    def evolve(self, t_increase, store_all = False, save_file=False, out_file = 'time_evolution.csv'):
        '''
        Algorithm to integrate the Schrodinger equation up to time t_final
        '''
            
        if t_increase<0:
            print('Error. Time increment must be positive')
            return
        
        N_steps = int(np.floor((t_increase)/self.dt)+1) #Number of steps to reach t_final from current time
        
        #Initialization: Obtain psi_1 from psi_0 using a Runge-Kutta method
        sol = solve_ivp(self.diff_RK, [0, self.dt], self.psi_xy_old.flatten())
        self.psi_xy = sol.y[:,-1].reshape(self.Ny,self.Nx)
        if store_all:
            self.psi_xy_all = np.concatenate((self.psi_xy_all, self.psi_xy.reshape(1,self.Ny, self.Nx)), axis=0)
    
        #For each time step:
        for i in range(1, N_steps):
            print('Time step: {}/{}'.format(i, N_steps), end='\r')
        # 1. Calculate nabla^2psi(x,t) by:
            # a) Obtain psi(k,t) using FFT
            psi_k = self.compute_psi_k_from_psi_x(self.psi_xy.copy())
            
            # b) Multiply psi(k,t) by hbar^2 k^2/(2m)
            nabla_psi_k = -psi_k*(self.kx**2 + self.ky**2)
            
            # c) Obtain the kinetic part in the x space using the inverse FFT
            nabla_psi_xy = -self.hbar**2/(2*self.m)*self.compute_psi_x_from_psi_k(nabla_psi_k)

        # 2. Add V(x)psi(x,t) to obtain H*psi(x,t)
            H_psi = nabla_psi_xy +  self.V*self.psi_xy
            
                
        # 3. Calculate psi(x, t+dt) = psi(x, t-dt) - 2i/h*dt*H*psi(x,t) 
            aux = self.psi_xy.copy() # Store value to set it as the old value later
            self.psi_xy = self.psi_xy_old - 2*1j/self.hbar*self.dt*H_psi
            self.psi_xy_old = aux #Set old value of psi
           
            if store_all:
                self.psi_xy_all = np.concatenate((self.psi_xy_all, self.psi_xy.reshape(1,self.Ny, self.Nx)), axis=0)
            if save_file:
                if i==1:
                    with open(out_file,'wb') as f:
                        np.savetxt(f, self.psi_xy.flatten(), delimiter=',', newline='\n')
                elif i%3==0:
                    with open(out_file,'a') as f:
                        np.savetxt(f, self.psi_xy.flatten(), delimiter=',', newline='\n')
        # Increase final time
        self.t += t_increase        

        


        
        
def calc_norm(phi, xmin=-5, xmax=5, ymin=-10, ymax = 10):
    '''
    Calculates the norm of an eigenfunction
    Args:
        phi (np.array): wave function values
        xmin (float): minimum value of x
        xmax (float): maximum value of x
        ymin (float): minimum value of y
        ymax (float): maximum value of y
        n_points (int): Number of grid points of x
    Returns:
        (float): norm of phi(x)
    '''
    (Ny,Nx) = phi.shape
    h1 = (xmax - xmin)/Nx
    h2 = (ymax - ymin)/Ny
    return 1./np.sqrt(np.sum(np.conjugate(phi)*phi*h1*h2))
     
def empirical_energy(phi, potential, xmin=-10, xmax = 10,
                       ymin=-10, ymax = 10, hbar=1, m=1):
    '''
    Calculates empirical energy for 2D potentials
    Args:
      phi (np.array): Wavefunction
      potential (np.array): Potentials V(x,y)
      xmin (int): minimum value of x
      xmax (int): maximum value of x
      ymin (int): minimum value of y
      ymax (int): maximum value of y
      n_points (int): number of points in the grid
      hbar (float): h bar
      m (float): mass
    Returns:
      energy (np.array): mean empirical energy for each sample
    '''
    (_,Ny,Nx) = phi.shape
    h1 = (xmax - xmin)/Nx
    h2 = (ymax - ymin)/Ny
    phi = np.asarray(phi)
    if phi.shape!= potential.shape:
        phi = np.reshape(phi, potential.shape)

    # We first calculate the second derivative of phi
    # derivative x
    phir = phi.copy()
    phir[:,:,0] = 0
    phir[:,:,1:] = phi[:,:,:-1]
    phil = phi.copy()
    phil[:,:,-1] = 0
    phil[:,:,:-1] = phi[:,:,1:]
    deriv_x = (phir - 2*phi + phil)/(h1*h2)

    # derivative y
    phir = phi.copy()
    phir[:,0,:] = 0
    phir[:,1:,:] = phi[:,:-1,:]
    phil = phi.copy()
    phil[:,-1,:] = 0
    phil[:,:-1,:] = phi[:,1:,:]
    deriv_y = (phir - 2*phi + phil)/(h1*h2)

    # Now we calculate the mean energy
    energy = np.sum((-hbar*hbar/(2*m)*np.conjugate(phi)*(deriv_x + deriv_y) + potential*(np.conjugate(phi)*phi))*h1*h2, axis=(1,2))
    return energy

        