import pandas as pd
import gensim
from sklearn.cross_validation import train_test_split
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, Imputer
from ndcg import *

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
        print result
    return score/float(count)

def main():
    print "============== XGB with bytecup validation =============="
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
    gbm = XGBClassifier()
    gbm.set_params(**{'n_estimators':2000, 'learning_rate':0.9, 'subsample':0.5, 'colsample_bytree':0.5, 'max_depth':20})
    gbm.fit(train_data, train_label,
        eval_set = [(train_data, train_label), (test_data, test_label)],
        eval_metric = 'auc', verbose=True)
    result = gbm.booster().get_fscore()
    se = pd.Series(result)
    important_feature = pd.to_numeric(se[lambda x: x>8].index)

    data_important = data[important_feature]
    gbm.fit(data_important, label, eval_metric = 'auc', verbose=True)

    left = validate
    right1 = user_new
    right2 = question_new
    left.columns = validate.columns
    right1_columns = [left.columns[1]]
    right1_columns.extend(list(range(right1.columns.size-1)))
    right1.columns = right1_columns
    right2_columns = [left.columns[0]]
    right2_columns.extend(list(range(right2.columns.size-1)))
    right2.columns = right2_columns
    left = pd.merge(left, right1, how='left', on=[validate.columns[1]])
    left = pd.merge(left, right2, how='left', on=[validate.columns[0]])
    data = left.drop(validate.columns, axis=1)
    label = left['label']
    impute = Imputer()
    data = pd.DataFrame(impute.fit_transform(data), index=data.index)
    std = StandardScaler()
    data = pd.DataFrame(std.fit_transform(data), index=data.index)
    data_important = data[important_feature]
    pred_label = pd.Series(gbm.predict_proba(data_important)[:,1], index=data_important.index)
    df_pred = pd.concat([validate[['qid','uid']].ix[label.index,:],
                         pd.DataFrame(pred_label, columns=['label'])], axis=1)
    df_pred.to_csv('temp.csv', index=None, encoding='utf-8')
    print "temp.csv is generated with probabilities for bytecup validation set."

if __name__ == '__main__':
    main()
