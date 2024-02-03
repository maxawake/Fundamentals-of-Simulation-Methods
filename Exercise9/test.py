import matplotlib.pyplot as plt
import numpy as np

c_sound = 1


def getConserved(rho, vx, P, gamma, vol):
    Mass = rho * vol
    Momx = rho * vx * vol

    return Mass, Momx


def getPrimitive(Mass, Momx, Energy, gamma, vol):
    rho = Mass / vol
    vx = Momx / rho / vol

    P = c_sound**2 * rho

    return rho, vx, P


def getGradient(f, dx):
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    f_dx = (np.roll(f, R, axis=0) - np.roll(f, L, axis=0)) / (2 * dx)

    return f_dx


def slopeLimit(f, dx, f_dx):
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    f_dx = np.maximum(0.0, np.minimum(1.0, ((f - np.roll(f, L, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx
    f_dx = np.maximum(0.0, np.minimum(1.0, (-(f - np.roll(f, R, axis=0)) / dx) / (f_dx + 1.0e-8 * (f_dx == 0)))) * f_dx

    return f_dx


def extrapolateInSpaceToFace(f, f_dx, dx):
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    f_XL = f - f_dx * dx / 2
    f_XL = np.roll(f_XL, R, axis=0)
    f_XR = f + f_dx * dx / 2

    return f_XL, f_XR


def applyFluxes(F, flux_F_X, dx, dt):
    # directions for np.roll()
    R = -1  # right
    L = 1  # left

    # update solution
    F += -dt * dx * flux_F_X
    F += dt * dx * np.roll(flux_F_X, L, axis=0)

    return F


def getFlux(rho_L, rho_R, vx_L, vx_R, P_L, P_R, gamma):
    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)

    P_star = c_sound**2 * rho_star

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star

    # find wavespeeds
    # C_L = c_sound + np.abs(vx_L)
    # C_R = c_sound + np.abs(vx_R)
    # C = np.maximum(C_L, C_R)

    # # add stabilizing diffusive term
    # flux_Mass -= C * 0.5 * (rho_L - rho_R)
    # flux_Momx -= C * 0.5 * (rho_L * vx_L - rho_R * vx_R)

    return flux_Mass, flux_Momx


result = []


def main():
    """Finite Volume simulation"""

    gamma = 1

    tEnd = 100
    courant_fac = 0.4
    tOut = 0.1
    useSlopeLimiting = True
    plotRealTime = True

    x = np.linspace(-100, 100, 200)
    dx = x[1] - x[0]
    vol = dx
    c_sound = 1
    t = 0

    dt = courant_fac * dx / c_sound

    rho = 1 + np.exp(-(x**2) / 100)
    vx = np.zeros(x.shape)
    P = c_sound**2 * rho

    # Get conserved variables
    Mass, Momx = getConserved(rho, vx, P, gamma, vol)

    # prep figure
    fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    # Simulation Main Loop
    while t < tEnd:
        Energy = None

        # get Primitive variables
        rho, vx, P = getPrimitive(Mass, Momx, Energy, gamma, vol)

        # get time step (CFL) = dx / max signal speed
        dt = courant_fac * np.min(dx / (c_sound + np.sqrt(vx**2)))

        plotThisTurn = False
        if t + dt > outputCount * tOut:
            dt = outputCount * tOut - t
            plotThisTurn = True

        # calculate gradients
        rho_dx = getGradient(rho, dx)
        vx_dx = getGradient(vx, dx)
        P_dx = getGradient(P, dx)

        # slope limit gradients
        if useSlopeLimiting:
            rho_dx = slopeLimit(rho, dx, rho_dx)
            vx_dx = slopeLimit(vx, dx, vx_dx)
            P_dx = slopeLimit(P, dx, P_dx)

        # extrapolate half-step in time
        rho_prime = rho - 0.5 * dt * (vx * rho_dx + rho * vx_dx)
        vx_prime = vx - 0.5 * dt * (vx * vx_dx + (1 / rho) * P_dx)
        P_prime = P - 0.5 * dt * (gamma * P * (vx_dx) + vx * P_dx)

        # extrapolate in space to face centers
        rho_XL, rho_XR = extrapolateInSpaceToFace(rho_prime, rho_dx, dx)
        vx_XL, vx_XR = extrapolateInSpaceToFace(vx_prime, vx_dx, dx)
        P_XL, P_XR = extrapolateInSpaceToFace(P_prime, P_dx, dx)

        # compute fluxes (local Lax-Friedrichs/Rusanov)
        flux_Mass_X, flux_Momx_X = getFlux(rho_XL, rho_XR, vx_XL, vx_XR, P_XL, P_XR, gamma)

        # update solution
        Mass = applyFluxes(Mass, flux_Mass_X, dx, dt)
        Momx = applyFluxes(Momx, flux_Momx_X, dx, dt)

        # update time
        t += dt

        result.append(rho)

        # # plot in real time
        if (plotRealTime and plotThisTurn) or (t >= tEnd):
            plt.cla()
            plt.plot(rho)
            plt.ylim(0, 3)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.pause(0.001)
            outputCount += 1

    # Save figure
    plt.savefig("finitevolume.png", dpi=240)
    plt.show()

    return 0


if __name__ == "__main__":
    main()
