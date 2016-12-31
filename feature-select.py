import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import cross_val_score

from sklearn import preprocessing


def load_dataset(shared_path, design_path):
    # load dataset
    shared = pd.read_table(shared_path)
    design = pd.read_table(design_path, header=None)

    # TODO: use LabelEncoder instead of custom function
    # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html

    # find the unique labels from the design files 2nd column
    labels = design[1].unique()
    # assign interger numbers to each labels and create map
    maps = dict(zip(labels, list(range(len(labels)))))
    # map these integer values to the the group names
    design_mapped = pd.Series(design[1].map(maps).values, index=design[0].values, dtype=int)
    # map these integer vaues to shared file to create the output column
    y = shared.iloc[:, 1].map(design_mapped)
    # remove the first 3 columns to get the value of X
    X = shared.drop(['Group', 'label', 'numOtus'], axis=1)
    print('Number of input features: {}'.format(len(X.keys())))
    return X, y


def preprocess_data(X, y, std_percent, corr_threshold):
    # std. deviation based pruning

    # calcuate std. dev. for all of the feature columns
    stddevs = X.std()
    # the threshold is set as a small percentage of the maximum std. deviation.
    # this makes the threshold automatically tuneable depending on the highest and lowest variations between
    # feature columns
    std_threshold = stddevs.max() * std_percent
    filtered = stddevs[stddevs > std_threshold]
    X = X[filtered.keys()]
    print('Number of input features after standard deviation based pruning: {}'.format(len(X.keys())))

    # person corr. based univariate feature selection
    pearson_corr = X.corr()
    keys = pearson_corr.keys()
    dupes = []  # if a pair has high correlation, put the second item in the pair in the dupe list
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            p, q = keys[i], keys[j]
            if q in dupes: continue
            # print(p, q, pearsons[p][q])
            if pearson_corr[p][q] > corr_threshold or pearson_corr[p][q] < -corr_threshold:
                dupes.append(q)
    X = X.drop(dupes, axis=1)
    print('Number of input features after pearson\'s correlation check with threshold value {}: {}' \
          .format(corr_threshold, len(X.keys())))
    return X, y


def select_features_univariate(X, y, percentile=10):
    # univariate feature selection with chi square test
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import chi2

    X_new = pd.DataFrame(SelectPercentile(score_func=chi2, percentile=percentile).fit_transform(X, y))

    # calculate the hash value for each of the orignal columns
    hashes = {}
    features_selected = []
    for c in X.keys(): hashes[hash(tuple(X[c].values))] = c
    # use the old hash values to find the column names now
    for c in X_new.keys():
        features_selected.append(hashes[hash(tuple(X_new[c].values))])
    print('Here are the top {}% (out of {}) features from univariate chi-square test:\n{}'
          .format(percentile,
                  len(X.keys()),
                  features_selected))


def select_features_svm_rfe(X, y, cross_val_folds):
    # recursive feature elimination with SVM
    # http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html

    svc = SVC(kernel='linear', class_weight='balanced', C=0.025)
    cross_validator = StratifiedKFold(n_splits=cross_val_folds, shuffle=True)
    # cross_validator = KFold(n_splits=cross_val_folds)
    rfecv = RFECV(estimator=svc, step=1, cv=cross_validator, scoring='accuracy', verbose=0, n_jobs=-1)
    X_scaled = preprocessing.scale(X)
    rfecv.fit(X_scaled, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    ranking_svm_rfe = pd.Series(rfecv.ranking_, index=X.keys())
    ranking_svm_rfe.sort_values(inplace=True)
    print('Feature ranking with SVM-RFE\n:{}'.format(ranking_svm_rfe))

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def select_features_rforest_rfe(X, y, cross_val_folds, numforests):
    # recursive feature elimination with SVM
    # http://scikit-learn.org/stable/auto_examples/feature_selection/plot_rfe_with_cross_validation.html

    rfc = RandomForestClassifier(n_estimators=numforests, criterion='entropy',
                                 oob_score=True, verbose=1, class_weight='balanced')
    cross_validator = StratifiedKFold(n_splits=cross_val_folds, shuffle=True)
    # cross_validator = KFold(n_splits=cross_val_folds)
    rfecv = RFECV(estimator=rfc, step=1, cv=cross_validator, scoring='accuracy', verbose=0, n_jobs=-1)
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)
    ranking_svm_rfe = pd.Series(rfecv.ranking_, index=X.keys())
    ranking_svm_rfe.sort_values(inplace=True)
    print('Feature ranking with RandomForest-RFE\n:{}'.format(ranking_svm_rfe))

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


def select_features_rforest(X, y, numforests, percentile=10):
    # more experiment with radom forests
    rfc = RandomForestClassifier(n_estimators=numforests, criterion='entropy',
                                 oob_score=True, n_jobs=-1, verbose=1, class_weight='balanced')
    rfc.fit(X, y)
    ranking_rforest = pd.Series(rfc.feature_importances_, index=X.keys())
    ranking_rforest.sort_values(inplace=True, ascending=False)
    ranking_rforest_top = ranking_rforest[:int(len(ranking_rforest) / 100 * percentile)]
    print('Feature ranking with random forest (top {}%):\n{}'
          .format(percentile, ranking_rforest_top))


def test_classifiers(X, y):
    # scale the features first
    X_scaled = preprocessing.scale(X)
    classifiers = {
        'svmlinear': SVC(kernel='linear', class_weight='balanced', C=0.025),
        'randomforest': RandomForestClassifier(n_estimators=200, criterion='entropy', oob_score=True,
                                               class_weight='balanced'),
        'knn': KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1),
        'naivebayes': GaussianNB(),
        'mlp': MLPClassifier(alpha=1)
    }
    for name in classifiers.keys():
        scores = cross_val_score(classifiers[name], X_scaled, y, cv=5, n_jobs=-1)
        print('Mean accuracy with {} classifier: {}'.format(name, scores.mean()))


def select_feature_from_model(X, y, max_features):
    from sklearn.feature_selection import SelectFromModel

    X_scaled = pd.DataFrame(preprocessing.scale(X), columns=X.keys())
    classifier = SVC(kernel='linear', class_weight='balanced', C=0.025)
    sfm = SelectFromModel(classifier, threshold=0.05)
    sfm.fit(X_scaled, y)
    n_features = sfm.transform(X_scaled).shape[1]
    while n_features > max_features:  # set the max number of features to select
        sfm.threshold += 0.05
        X_transform = sfm.transform(X_scaled)
        n_features = X_transform.shape[1]
    X_final = pd.DataFrame(X_transform)

    hashes = {}
    features_selected = []
    for c in X_scaled.keys(): hashes[hash(tuple(X_scaled[c].values))] = c
    for c in X_final.keys():
        features_selected.append(hashes[hash(tuple(X_final[c].values))])
    print('Features selection by SelectFromModel: {}'.format(features_selected))


# if __name__ == '__main__':

# parameters
std_percent = 0.01  # should be 0.01 normally
corr_threshold = 0.8

X, y = load_dataset('datasets/WTmiceonly_final.shared', 'datasets/WTmiceonly_final.design')

# X, y = load_dataset('datasets/WTonly.subsample.0.03.shared', 'datasets/WTonly.time.design')
# X, y = load_dataset('datasets/HumanCRC.final.subsample.shared', 'datasets/HumanCRC.design')
# X, y = load_dataset('datasets/inpatient.final.an.0.03.subsample.avg.shared', 'datasets/inpatient.final.an.0.03.subsample.avg.design')

X, y = preprocess_data(X, y, std_percent, corr_threshold)

# select_features_univariate(X, y)
# select_features_rforest(X, y, numforests=200)
# select_features_svm_rfe(X, y, cross_val_folds=5)
# select_features_rforest_rfe(X, y, cross_val_folds=5, numforests=100)
# test_classifiers(X, y)
select_feature_from_model(X, y, 10)