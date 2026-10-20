import numpy as np
from astropy.table import Table

emb_dir = "/home/valonso/iac18_mhuertas_shared/valonso/astroPT/logs/astropt_20260612_autoregresive_optimized/embeddings_full/best_img-mean_spec-rank_final_iso_j-mean"
ids = np.load(emb_dir + "/ids.npy")
print("Number of IDs in embeddings_full/ids.npy:", len(ids))
print("First 10 IDs:", ids[:10])
print("Number of unique IDs:", len(np.unique(ids)))
print("Number of zero IDs:", np.sum(ids == 0))

# Load metadata to see intersection
metadata_path = "/home/valonso/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
catalog = Table.read(metadata_path)
meta_ids = np.array(catalog['targetid'], dtype=np.int64)
print("Number of IDs in catalog:", len(meta_ids))

intersection = np.intersect1d(ids, meta_ids)
print("Number of aligned samples (intersection):", len(intersection))
