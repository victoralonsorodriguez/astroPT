import sys
from astropy.table import Table

fits_path = "/home/valonso/iac18_mhuertas_shared/valonso/AGNs/StellarMass_Compilation_V2.fits"
tab = Table.read(fits_path)
print("FITS Columns:", tab.colnames)
print("First row samples:")
print(tab[:3])
