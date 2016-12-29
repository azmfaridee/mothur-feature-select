import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# parameters
std_percent = 0.01  # should be 0.01 normally
corr_threshold = 0.8


def load_dataset(shread_path, design_path, design_int_maps):
    # load dataset
    shared = pd.read_table('WTmiceonly_final.shared')
    design = pd.read_table('WTmiceonly_final.design', header=None)

    shared.drop(['Group', 'label', 'numOtus'], axis=1, inplace=True)
    for k in design_int_maps.keys():
        design.loc[design[1] == k, 1] = design_int_maps[k]

    # design.loc[design[1] == 'Before', 1] = 0
    # design.loc[design[1] == 'After1', 1] = 1
    # design.loc[design[1] == 'After2', 1] = 2
    # design.loc[design[1] == 'After3', 1] = 3
    print('Number of input features: {}'.format(len(shared.keys())))
    return shared, design

def preprocess_data(shared, design, std_percent, corr_threshold):
    # dataset is ready for analysis
    # first remove almost null columns
    stddevs = shared.std()
    std_threshold = stddevs.max() * std_percent
    filtered = stddevs[stddevs > std_threshold]
    shared = shared[filtered.keys()]
    print('Number of input features after standard deviation based pruning: {}'.format(len(shared.keys())))

    pearson_corr = shared.corr()

    keys = pearson_corr.keys()
    dupes = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = keys[i], keys[j]
            if y in dupes: continue
            # print(x, y, pearsons[x][y])
            if pearson_corr[x][y] > corr_threshold or pearson_corr[x][y] < -corr_threshold:
                dupes.append(y)
    shared = shared.drop(dupes, axis=1)
    print('Number of input features after pearson\'s correlation check with threshold value {}: {}' \
          .format(corr_threshold, len(shared.keys())))
    return shared, design


shared, design = load_dataset('WTmiceonly_final.shared', 'WTmiceonly_final.design', \
                              {'Before': 0, 'After1': 1, 'After2': 2, 'After3': 3})
shared, design = preprocess_data(shared, design, std_percent, corr_threshold)
# assign output for each row
# shared['output'] = design[1]
X, y = shared.values, np.array(design[1].values, dtype=int)

# univariate feature selection with chi square test
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2
X_new = pd.DataFrame(SelectPercentile(score_func=chi2, percentile=5).fit_transform(X, y))
