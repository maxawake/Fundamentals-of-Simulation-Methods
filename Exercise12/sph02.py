import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.special import gamma

# two different kernels, a Gaussian and a spline


# Gaussian kernel
def W(x, y, z, h):
    """
    Gausssian kernel in 3D
        x,y,z     list of positions in x, y, z
        h         smoothing length

        return    smoothing function
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    return (1.0 / (h * np.sqrt(np.pi))) ** 3 * np.exp(-(r**2) / h**2)


# derivative of Gaussian kernel
def grad_W(x, y, z, h):
    """
    Gradient of the Gausssian kernel W
    x,y,z     list of positions in x, y, z
    h         smoothing length
    wx,wy,wz  gradient of W
    """

    r = np.sqrt(x**2 + y**2 + z**2)

    n = -2.0 * np.exp(-(r**2) / h**2) / h**5 / (np.pi) ** (3.0 / 2.0)
    dwx = n * x
    dwy = n * y
    dwz = n * z

    return dwx, dwy, dwz


def compute_pairwise_distances(ri, rj):
    """
    compute pairwise separations between 2 sets of coordinates
    ri    is an M x 3 matrix of positions
    rj    is an N x 3 matrix of positions
    dx, dy, dz   are M x N matrices of separations
    """

    M = ri.shape[0]
    N = rj.shape[0]

    # positions ri = (x,y,z)
    rix = ri[:, 0].reshape((M, 1))
    riy = ri[:, 1].reshape((M, 1))
    riz = ri[:, 2].reshape((M, 1))

    # other set of points positions rj = (x,y,z)
    rjx = rj[:, 0].reshape((N, 1))
    rjy = rj[:, 1].reshape((N, 1))
    rjz = rj[:, 2].reshape((N, 1))

    # matrices that store all pairwise particle separations: r_i - r_j
    dx = rix - rjx.T
    dy = riy - rjy.T
    dz = riz - rjz.T

    return dx, dy, dz


def compute_density(r, pos, m, h):
    """
    Compute density at sampling loctions from SPH particle distribution
    r     is an M x 3 matrix of sampling locations
    pos   is an N x 3 matrix of SPH particle positions
    m     is the particle mass
    h     is the smoothing length
    rho   is M x 1 vector of densities
    """

    M = r.shape[0]

    dx, dy, dz = compute_pairwise_distances(r, pos)

    rho = np.sum(m * W(dx, dy, dz, h), 1).reshape((M, 1))

    return rho


def compute_pressure(rho, k, n):
    """
    equation of state
    rho   vector of densities
    k     constant for the equation of state
    n     polytropic index
    """

    return k * rho ** (1 + 1 / n)


def compute_accelerations(pos, vel, m, h, k, n, Fext, nu):
    """
    calculate the acceleration on each SPH particle
    pos   is an N x 3 matrix of positions
    vel   is an N x 3 matrix of velocities
    m     is the particle mass
    h     is the smoothing length
    k     constant for the equation of state
    n     polytropic index
    Fext  external force constant
    nu    viscosity
    a     is N x 3 matrix of accelerations
    """

    N = pos.shape[0]

    # Calculate densities at the position of the particles
    rho = compute_density(pos, pos, m, h)

    # Get the pressures
    P = compute_pressure(rho, k, n)
    print(P.shape)

    # Get pairwise distances and gradients
    dx, dy, dz = compute_pairwise_distances(pos, pos)
    dWx, dWy, dWz = grad_W(dx, dy, dz, h)

    tmp = P / rho**2
    tmp2 = tmp + tmp.T
    print(tmp2.shape)

    # Add Pressure contribution to accelerations
    ax = -np.sum(m * (P / rho**2) * dWx, axis=0)
    ay = -np.sum(m * (P / rho**2) * dWy, axis=0)
    az = -np.sum(m * (P / rho**2) * dWz, axis=0)

    # pack together the acceleration components
    a = np.hstack((ax, ay, az))

    # Add external potential force
    a -= Fext * pos

    # Add viscosity
    a -= nu * vel

    return a


def run_simulation(tEnd=12, dt=0.04, N=400, h=0.1, nu=1):
    """parameters for the main simulation funcion
    tEnd   : end time
    dt     : time step
    N      : number of particles
    h      : globally constant smoothing length
    nu     : artificial viscosity
    """

    # Main simulation parameters
    t = 0  # current time of the simulation
    M = 2  # star mass
    R = 0.75  # star radius
    k = 0.1  # equation of state constant
    n = 1  # polytropic index

    # Generate Initial Conditions
    np.random.seed(42)  # set the random number generator seed

    Fext = (
        2 * k * (1 + n) * np.pi ** (-3 / (2 * n)) * (M * gamma(5 / 2 + n) / R**3 / gamma(1 + n)) ** (1 / n) / R**2
    )  # ~ 2.01
    m = M / N  # single particle mass
    pos = np.random.randn(N, 3)  # randomly selected positions and velocities
    vel = np.zeros(pos.shape)

    # calculate initial gravitational accelerations
    acc = compute_accelerations(pos, vel, m, h, k, n, Fext, nu)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # Simulation Main Loop
    for i in range(Nt):
        print(
            "time step ",
            i,
        )

        # kick-drift-kick
        # Kick
        vel += 0.5 * acc * dt

        # Drift
        pos += vel * dt

        # Calculate new acceleration
        acc = compute_accelerations(pos, vel, m, h, k, n, Fext, nu)

        # Kick
        vel += 0.5 * acc * dt

        # update time
        t += dt

        # get density for plotting
        rho = compute_density(pos, pos, m, h)
        plt.imshow(rho.reshape((20, 20)), origin="lower", cmap="turbo")

        plt.show()

    # create final figure with positions of particles and density coloour coded
    # .....

    return 0


def main():
    """SPH setup"""

    run_simulation(tEnd=12.0, dt=0.04, N=400, h=0.1, nu=1.0)

    return 0


if __name__ == "__main__":
    main()
