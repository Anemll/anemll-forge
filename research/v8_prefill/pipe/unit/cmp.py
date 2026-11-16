import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
print(" ".join(f"{k}:{float(np.abs(a[k]-b[k]).mean()/np.abs(a[k]).mean()):.4f}" for k in sorted(a.files, key=lambda s: int(s[1:])) if k.startswith('y')))
