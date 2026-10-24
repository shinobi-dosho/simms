`pybdsf-1.14.1-native.gaul` is the unmodified 43-Gaussian ASCII catalogue
written by PyBDSF 1.14.1 for the user's CHI_OPH scan-9 third-round image.
Copied with the user's authorization on 2026-10-24 from Impala:

`/home/sphemakh/reductions/reduce-chiop/output-live-scan9/ascii-gaul-check/rnd_three-ascii.gaul`

Its header records the input image and detection reference frequency
(1.42001e9 Hz). All original measurement/error columns and negative RA
values are retained so regression tests exercise native output rather than
an artificially reduced `#format:` table. Tests use this local fixture;
no remote data access is needed. Spectral-index and polarization cases in
`pybdsf_native_tests.py` are synthetic extensions because this actual
single-frequency catalogue has neither column set.
