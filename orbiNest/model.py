import numpy as np
from numpy import pi

# ==========================
# ASTROPHYSICAL CONSTANTS
# ==========================

G = 6.67408e-11             # [m^3 kg^-1 s^-2] Gravitational constant
Msun = 1.98847e30           # [kg] Solar mass
AU = 1.495978707e11         # [m] Astronomical unit
d2s = 24 * 3600             # [s/day]
s2yr = d2s * 365.25         # [s/year]

# ==========================
# ORBITAL FUNCTIONS
# ==========================

def get_true_anomaly(P, e, M, tol=1e-6, max_iter=100):
    """
    Solve Kepler's equation for eccentric anomaly E and return true anomaly T.
    """
    E = M.copy()
    e = np.abs(e)
    for _ in range(max_iter):
        delta_E = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E -= delta_E
        if np.max(np.abs(delta_E)) < tol:
            break
    T = 2 * np.arctan2(
        np.sqrt(1 + e) * np.sin(E / 2),
        np.sqrt(1 - e) * np.cos(E / 2)
    )
    return T

def orbital_period(m0, m1, a_AU):
    """
    Kepler's Third Law: orbital period in days.
    """
    P_sec = 2 * pi * np.sqrt((a_AU * AU)**3 / (G * (m0 + m1) * Msun))
    return P_sec / d2s

def rv_amplitude(m0, m1, P, e, inclination=np.pi / 2):
    """
    Radial velocity semi-amplitude [km/s]
    """
    P_sec = P * d2s
    factor = (2 * pi / P_sec)**(1/3)
    amplitude = (
        m1 * Msun * factor *
        (G * (m0 + m1) * Msun)**(-2/3) *
        np.sin(inclination) / np.sqrt(1 - e**2)
    )
    return amplitude / 1000  # m/s to km/s

def rv_model(theta, t, T_ref=51544.):
    """
    Compute RV curve at times `t` given orbital parameters.

    Parameters in `theta`:
    - K: semi-amplitude [km/s]
    - P: period [days]
    - tau: phase (fraction of period since periastron)
    - e: eccentricity
    - w: argument of periastron [radians]
    - off: systemic velocity offset [km/s]
    """
    theta = np.atleast_2d(theta)

    K = theta[:, [0]]
    P = theta[:, [1]]
    tau = theta[:, [2]]
    e = np.abs(theta[:, [3]])
    w = theta[:, [4]]
    off = theta[:, [5]]

    frac_date = ((t - T_ref) / P) % 1
    M = ((frac_date - tau) * 2 * pi) % (2 * pi)
    T = get_true_anomaly(P, e, M)

    rv = K * (np.cos(w + T) + e * np.cos(w)) + off
    return rv
