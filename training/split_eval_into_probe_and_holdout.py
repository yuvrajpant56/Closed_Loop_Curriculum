# scripts/split_eval_into_probe_and_holdout.py
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/mmfs1/scratch/jacks.local/pkhanal2568/yuv_workshop_newnew_3b/local_data")
PROBE_SIZE = 100   # items per bucket used to drive the controller
SPLIT_SEED = 20260511  # fixed forever; record this in the paper

for n in [2, 3, 4, 5]:
    src = DATA_DIR / f"countdown_n{n}_test.parquet"
    df = pd.read_parquet(src)
    rng = np.random.default_rng(SPLIT_SEED + n)
    perm = rng.permutation(len(df))

    probe_idx = perm[:PROBE_SIZE]
    test_idx = perm[PROBE_SIZE:]

    df.iloc[probe_idx].reset_index(drop=True).to_parquet(
        DATA_DIR / f"countdown_n{n}_probe.parquet"
    )
    df.iloc[test_idx].reset_index(drop=True).to_parquet(
        DATA_DIR / f"countdown_n{n}_holdout.parquet"
    )
    print(f"n{n}: probe={len(probe_idx)}, holdout={len(test_idx)}")