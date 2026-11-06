import glob, os, sys
import numpy as np, pandas as pd

KINDS = ("pooled_wholebrain", "region_pooled", "region_crosssubject_metrics")
SKIP  = {"label", "region", "region_id", "index", "unnamed: 0", "roi", "n"}

def read_kind(d, kind):
    hits = sorted(glob.glob(os.path.join(d, kind + "_*.csv")))
    if not hits:
        return None
    df  = pd.read_csv(hits[0])
    num = df.select_dtypes(include=[np.number])
    num = num.drop(columns=[c for c in num.columns if c.lower() in SKIP], errors="ignore")
    if num.empty:
        return None
    # 1-row file -> the value itself; 86-row file -> mean across regions.
    if len(df) == 1:
        return {c: float(num[c].iloc[0]) for c in num.columns}
    return {c: float(np.nanmean(num[c])) for c in num.columns}

def summarize(root):
    folds = sorted(glob.glob(os.path.join(root, "fold_*")))
    dirs  = folds if folds else [root]
    print("=" * 78)
    print("%s   [%d dir(s)]" % (root, len(dirs)))
    for kind in KINDS:
        vals = [v for v in (read_kind(d, kind) for d in dirs) if v]
        if not vals:
            print("  %-30s -- missing --" % kind)
            continue
        print("  %s  (n=%d)" % (kind, len(vals)))
        for c in sorted({c for v in vals for c in v}):
            arr = np.array([v[c] for v in vals if c in v], float)
            if len(arr) > 1:
                print("      %-16s %.4f +/- %.4f" % (c, np.nanmean(arr), np.nanstd(arr)))
            else:
                print("      %-16s %.4f" % (c, arr[0]))

for pattern in sys.argv[1:]:
    for d in sorted(glob.glob(pattern)):
        if os.path.isdir(d):
            summarize(d)
