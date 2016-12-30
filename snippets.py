import pandas as pd
import numpy as np

shared = pd.read_table('WTmiceonly_final.shared')
design = pd.read_table('WTmiceonly_final.design', header=None)
labels = design[1].unique()
maps = dict(zip(labels, list(range(len(labels)))))
design_mapped = pd.Series(design[1].map(maps).values, index=design[0].values, dtype=int)

y = shared.iloc[:, 1].map(design_mapped)
print(y)