import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma


# Gaussian kernel
def W(x, y, z, h):
    """
    Gaussian kernel in 3D
    x,y,z     list of positions in x, y, z
    h         smoothing length

    return    smoothing function
    """

    r = np.sqrt(x**2 + y**2 + z**2)
    w = 1 / (np.pi * h**3) * np.exp(-(r**2) / h**2)

    return w


# derivative of Gaussian kernel
def grad_W(x, y, z, h):
    """
    Gradient of the Gausssian kernel W
    x,y,z     list of positions in x, y, z
    h         smoothing length
    wx,wy,wz  gradient of W
    """

    r = np.sqrt(x**2 + y**2 + z**2)
    dwx = -2 * x / (np.pi * h**5) * np.exp(-(r**2) / h**2)
    dwy = -2 * y / (np.pi * h**5) * np.exp(-(r**2) / h**2)
    dwz = -2 * z / (np.pi * h**5) * np.exp(-(r**2) / h**2)

    return dwx, dwy, dwz


def compute_pairwise_distances(ri, rj):
    """
    compute pairwise separations between 2 sets of coordinates
    ri    is an N x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are N x N matrices of separations
    """
    dx = ri[:, 0, np.newaxis] - rj[:, 0]
    dy = ri[:, 1, np.newaxis] - rj[:, 1]
    dz = ri[:, 2, np.newaxis] - rj[:, 2]

    return dx, dy, dz


def compute_density(r, pos, m, h):
    """
    Compute density at sampling locations from SPH particle distribution
    r     is an N x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M vector of densities
    """

    dist = compute_pairwise_distances(r, pos)
    rho = np.sum(m * W(dist[0], dist[1], dist[2], h), axis=1)

    return rho


def compute_number_neighbors(pos, h):
    """
    Compute number of neighbors within radius of 2*smoothing length
    """

    dx, dy, dz = compute_pairwise_distances(pos, pos)

    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    Nneigh = np.sum(distances < 2 * h, axis=1)

    return Nneigh


def main():
    """SPH setup"""

    # Simulation parameters
    N = 5000  # Number of particles
    M = 1.0  # total mass

    # Generate Initial Conditions
    np.random.seed(4711)  # set the random number generator seed

    m = M / N  # single particle mass
    pos = np.random.rand(N, 3)  # randomly selected positions and velocities

    resolution = 256
    lin = np.linspace(0, 1, resolution)
    x, y = np.meshgrid(lin, lin)
    r = np.array([x.flatten(), y.flatten(), np.zeros(x.shape).flatten()]).T

    # loop over different smoothing lengths
    for h in [0.02, 0.05, 0.1, 0.2, 0.3]:
        print("h=", h)

        # compute density at initial positions
        rho = compute_density(r, pos, m, h)

        # plot positions with density color coded
        plt.imshow(rho.reshape(resolution, resolution))
        plt.title(f"$h={h}$")
        plt.colorbar()
        plt.savefig("density_h" + str(h) + ".png")
        plt.show()

        # compute number of neighbors
        Nn = compute_number_neighbors(pos, h)

        # plot positions with Nn color coded
        plt.scatter(pos[:, 0], pos[:, 1], c=Nn, cmap="turbo")
        plt.savefig("scatter_h" + str(h) + ".png")
        plt.colorbar()
        plt.show()

        # make histogram of densities
        plt.hist(rho, bins=16, density=True)
        plt.savefig("histogram_densities" + str(h) + ".png")
        plt.show()

        # make histogram of number of neighbors
        plt.hist(Nn, bins=16, density=True)
        plt.savefig("histogram_neighbors" + str(h) + ".png")
        plt.show()

    return 0


if __name__ == "__main__":
    main()
