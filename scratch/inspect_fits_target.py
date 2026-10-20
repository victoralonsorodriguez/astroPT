import pandas as pd
from astropy.table import Table
import numpy as np

fits_path = "/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1_FILTERED.fits"
print("Loading catalog...")
tbl = Table.read(fits_path)
df = tbl.to_pandas()
print("Catalog loaded.")

target_id = 39633516366922547
row = df[df['TARGETID'] == target_id]
if len(row) == 0:
    # Try with index
    row = df[df['targetid'] == target_id]

if len(row) > 0:
    print("Found target!")
    row_data = row.iloc[0]
    cols = ['HALPHA_FLUX', 'HBETA_FLUX', 'OIII_5007_FLUX', 'OII_3726_FLUX', 'HALPHA_EW', 'HBETA_EW', 'OII_3726_EW']
    for col in cols:
        val = row_data.get(col, "MISSING_COL")
        print(f"  {col}: {val} (type: {type(val)})")
else:
    print("Target not found in catalog!")
