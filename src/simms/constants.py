# Frequently used constants are defined here
import numpy as np

PI = np.pi
C = 2.99792458e8
# fwhm scale factor: FWHM = FWHM_scale_fact * sigma
FWHM_scale_fact = 2 * np.sqrt(2 * np.log(2))
# Convert a Gaussian FWHM (an angle on the sky) to the uv-plane shape scale the
# kernel uses. The kernel envelope is exp(-emaj**2 * (u_lambda**2 + ...)); the
# Fourier transform of an image-plane Gaussian of FWHM theta, i.e.
# sigma = theta / FWHM_scale_fact, is exp(-2 pi**2 sigma**2 u_lambda**2), so
# emaj = theta * pi / (2 sqrt(ln 2)).
FWHM_TO_GAUSS_SCALE = np.pi / (2 * np.sqrt(np.log(2)))
# Earth's semi major axis.
earth_emaj = 6378137.0  # [m]
# Earth's first numerical eccentricity
esq = 0.00669437999014
# Earth's sidereal rotation rate [rad/s]. Sets how fast the uvw of a tracking
# baseline turn, and hence the fringe rate that time smearing integrates over.
OMEGA_EARTH = 7.292115e-5
