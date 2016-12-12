import gensim
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.cross_validation import train_test_split
from utils import *
from ndcg import *
from sklearn import pipeline, metrics, grid_search


def gini(y_true, y_pred):
    """ Simple implementation of the (normalized) gini score in numpy. 
        Fully vectorized, no python loops, zips, etc. Significantly
        (>30x) faster than previous implementions
        
        Credit: https://www.kaggle.com/jpopham91/
    """

    # check and get number of samples
    assert y_true.shape == y_pred.shape
    n_samples = y_true.shape[0]
    
    # sort rows on prediction column 
    # (from largest to smallest)
    arr = np.array([y_true, y_pred]).transpose()
    true_order = arr[arr[:,0].argsort()][::-1,0]
    pred_order = arr[arr[:,1].argsort()][::-1,0]
    
    # get Lorenz curves
    L_true = np.cumsum(true_order) / np.sum(true_order)
    L_pred = np.cumsum(pred_order) / np.sum(pred_order)
    L_ones = np.linspace(0, 1, n_samples)
    
    # get Gini coefficients (area between curves)
    G_true = np.sum(L_ones - L_true)
    G_pred = np.sum(L_ones - L_pred)
    
    # normalize to true Gini coefficient
    return G_pred/G_true
    
    
def normalized_gini(y_true, y_pred):
    ng = gini(y_true, y_pred)/gini(y_true, y_true)
    return ng


def fit(train, target):
    # set up pipeline
    est = pipeline.Pipeline([
            ('xgb', xgb.XGBRegressor(silent=True)),
        ])
        
    # create param grid for grid search
    params = {
        'xgb__learning_rate': [0.003, 0.005, 0.01, ],
        'xgb__min_child_weight': [5, 6, 7, ],
        'xgb__subsample': [0.5, 0.7, 0.9, ],
        'xgb__colsample_bytree': [0.5, 0.7, 0.9, ],
        'xgb__max_depth': [1, 3, 5, 7, 9, 11, ],
        'xgb__n_estimators': [10, 50, 100, ],
        }

    # set up scoring mechanism
    gini_scorer = metrics.make_scorer(normalized_gini, greater_is_better=True)
    
    # initialize gridsearch
    gridsearch = grid_search.RandomizedSearchCV(
        estimator=est,
        param_distributions=params,
        scoring=gini_scorer,
        verbose=10,
        n_jobs=-1,
        cv=3,
        n_iter=3,
        )
        
    # fit gridsearch
    gridsearch.fit(train, target)
    print('Best score: %.3f' % gridsearch.best_score_)
    print('Best params:')
    for k, v in sorted(gridsearch.best_params_.items()):
        print("\t%s: %r" % (k, v))
        
    # get best estimator
    return gridsearch.best_estimator_


def predict(est, test):
    y_pred = est.predict(test)
    y_pred.to_csv('submission.csv')
    print('Predictions saved to submission.csv')
    
def transform(series, dimension=100):
    row = series.index
    sentenses = series.str.split('/').dropna()
    model = gensim.models.Word2Vec(sentenses, workers=4, min_count=1, size=dimension)
    out = []
    for sentence in sentenses:
        result = np.zeros(dimension)
        count = 0
        for word in sentence:
            result += model[word]
            count +=1
        result = result / float(count)
        out.append(result)
    df = pd.DataFrame(out , index=sentenses.index).reindex(row).fillna(0)
    return df

def evaluate(df_true, df_pred, column_name='qid', label_name='label', method=0):
    keys = df_true.groupby(column_name).groups
    score = 0.0
    count = 0

    for key in keys:
        true = df_true[df_true[column_name]==key][label_name]
        pred = df_pred[df_pred[column_name]==key][label_name]
        index = pred.sort_values(ascending=False).index
        r = true.reindex(index).tolist()
        ndcg5 = ndcg_at_k(r, 5, method)
        ndcg10 = ndcg_at_k(r, 10, method)
        result = (ndcg5+ndcg10)/2.0
        score += result
        count += 1
    return score/float(count)

if __name__ == '__main__':
    print "=============== XGB and Gini Score with local validation ==============="
    invite = pd.read_csv('data/invited_info_train.txt', sep='\t', header=None)
    question = pd.read_csv('data/question_info.txt', sep='\t', header=None)
    user = pd.read_csv('data/user_info.txt', sep='\t', header=None)
    validate = pd.read_csv('data/validate_nolabel.txt', sep=',')
    question_new = pd.concat([question.drop([2,3], axis=1), transform(question[2]), transform(question[3])], axis=1)
    user_new = pd.concat([user.drop([1,2,3], axis=1), transform(user[1], 50),
                          transform(user[2]), transform(user[3])], axis=1)

    left = invite
    right1 = user_new
    right2 = question_new
    left.columns = validate.columns
    right1_columns = [left.columns[1]]
    right1_columns.extend(list(range(right1.columns.size-1)))
    right1.columns = right1_columns
    right2_columns = [left.columns[0]]
    right2_columns.extend(list(range(right2.columns.size-1)))
    right2.columns = right2_columns
    left = pd.merge(left, right1, how='left', on=[invite.columns[1]])
    left = pd.merge(left, right2, how='left', on=[invite.columns[0]])
    data = left.drop(invite.columns, axis=1)
    label = left[invite.columns[2]]    
    impute = Imputer()
    data = pd.DataFrame(impute.fit_transform(data), index=data.index)
    std = StandardScaler()
    data = pd.DataFrame(std.fit_transform(data), index=data.index)

    train_data, test_data, train_label, test_label = train_test_split(data, label, random_state=456,
                                                                  test_size=0.2, stratify=label)

    # load data
    train = train_data
    test = test_data
    
    target = train_label

    # preprocess categorical features and convert to numpy arrays
    X_train = train_data
    y_train = train_label
    X_test = test_data
    
    # randomize train set
    idx = np.random.permutation(len(train))
    X_train = X_train[idx]
    y_train = y_train[idx]
    
    # fit model
    est = fit(X_train, y_train)
    
    # generate predictions
    pred_label = est.predict(test)
    df_true = pd.concat([invite[['qid','uid']].ix[test_label.index,:],
                     pd.DataFrame(test_label, columns=['label'])], axis=1)
    df_pred = pd.concat([invite[['qid','uid']].ix[test_label.index,:],
                     pd.DataFrame(pred_label, columns=['label'])], axis=1)
    r = evaluate(test_label, df_pred, method=1)
    print "NDCG score with validation set (80% Training set: 20% Test set): ", r
