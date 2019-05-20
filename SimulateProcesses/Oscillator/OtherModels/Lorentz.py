import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

def discrete_Lorenz(steps, initial, sigma, beta, rho, dt):
	states = np.zeros((steps,3))
	states[0,:] = initial
	for s in range(1,steps):

		old = states[s-1]
		new = np.zeros(3)
		new[0] = dt*sigma*old[1] - dt*sigma*old[0] + old[0]
		new[1] = (rho*old[0] - old[0]*old[2] - old[1])*dt + old[1]
		new[2] = (old[0]*old[1] - (beta)*old[2])*dt + old[2]
		
		states[s,:] = new

	return states

if __name__ == "__main__":
	initial = np.random.rand(3)
	sigma = 10
	beta = 8/3
	rho = 28
	dt = 0.005
	states = discrete_Lorenz(5000, initial, sigma, beta, rho, dt)

	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(states[:,0],states[:,1],states[:,2])
	plt.show()
