from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score

import numpy as np
import gensim
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


class VotingClassifierMethod(object):
    '''A wrapper for sklearn voting classifier ensemble method
       Voting classfier class with soft voting.
       It takes weight for each classifer'''

    def __init__(self, weights):
        if len(weights) != 7:
            raise 'Weights array should be size 7'
        self.weights = weights
        self.init_classifiers()

    def init_classifiers(self, n_estimators=10, max_depth=10, learning_rate=0.1):
        '''n_jobs is set to -1 to make use of multiple cores'''

        self.clf1 = LogisticRegression(n_jobs=-1)
        self.clf2 = RandomForestClassifier(n_estimators=n_estimators, random_state=1, max_depth=max_depth, n_jobs=-1)
        self.clf3 = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)
        self.clf4 = KNeighborsClassifier(n_jobs=-1)
        self.clf5 = GradientBoostingClassifier(n_estimators=n_estimators,
                                               learning_rate=learning_rate, max_depth=max_depth, random_state=0)
        self.clf6 = ExtraTreesClassifier(n_estimators=n_estimators,
                                         random_state=1, max_depth=max_depth, n_jobs=-1)
        self.clf7 = XGBClassifier(n_estimators=n_estimators,
                                  max_depth=max_depth, learning_rate=learning_rate)
        self.classifier = VotingClassifier(estimators=[('lr', self.clf1),
                                                       ('rf', self.clf2),
                                                       ('ada', self.clf3),
                                                       ('knn', self.clf4),
                                                       ('gbc', self.clf5),
                                                       ('etc', self.clf6),
                                                       ('xgb', self.clf7)],
                                                        voting='soft',
                                                        weights=self.weights)
        print 'Classifier created'

    def grid_search(self, n_estimators, learning_rates, max_depths, x, y):
        '''Grid search for n_estimators, learning_rate and max_depth'''

        for estimator in n_estimators:
            for depth in max_depths:
                for learning_rate in learning_rates:
                    self.init_classifiers(estimator, depth, learning_rate)
                    scores = cross_val_score(self.classifier, x, y, cv=5, scoring='accuracy', n_jobs=-1)
                    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    def print_cross_validation_score(self, x, y):
        '''Selected classifiers cross_validation scores'''

        for clf, label in zip(
                [self.clf1, self.clf2, self.clf3, self.clf4, self.clf5, self.clf6, self.clf7],
                ['lr', 'rf', 'ada', 'knn', 'gbc', 'etc', 'xgb']):
            scores = cross_val_score(clf, x, y, cv=5, scoring='accuracy', n_jobs=-1)
            print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    def fit(self, x, y):
        self.classifier.fit(x, y)

    def predict(self, x):
        return self.classifier.predict_proba(x)


if __name__ == '__main__':
    print "============= Voting Classifier with bytecup validation ==============="
    import pandas as pd
    invite = pd.read_csv('data/invited_info_train.txt', sep='\t', header=None)[:100]
    question = pd.read_csv('data/question_info.txt', sep='\t', header=None)
    user = pd.read_csv('data/user_info.txt', sep='\t', header=None)
    validate = pd.read_csv('data/validate_nolabel.txt', sep=',')

    # formulate the feature
    question_new = pd.concat([question.drop([2, 3], axis=1), transform(question[2]), transform(question[3])], axis=1)
    user_new = pd.concat([user.drop([1, 2, 3], axis=1), transform(user[1], 50),
                          transform(user[2]), transform(user[3])], axis=1)

    left = invite
    right1 = user_new
    right2 = question_new
    left.columns = validate.columns
    right1_columns = [left.columns[1]]
    right1_columns.extend(list(range(right1.columns.size - 1)))
    right1.columns = right1_columns
    right2_columns = [left.columns[0]]
    right2_columns.extend(list(range(right2.columns.size - 1)))
    right2.columns = right2_columns
    left = pd.merge(left, right1, how='left', on=[invite.columns[1]])
    left = pd.merge(left, right2, how='left', on=[invite.columns[0]])
    label = left[invite.columns[2]]
    data = left.drop(invite.columns, axis=1)
    del left, right2, right1
    # deal with missing data and normalization
    from sklearn.preprocessing import StandardScaler, Imputer

    impute = Imputer()
    data = pd.DataFrame(impute.fit_transform(data), index=data.index)
    std = StandardScaler()
    data = pd.DataFrame(std.fit_transform(data), index=data.index)
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import f_classif

    model = SelectKBest(f_classif, k=250)
    new_data = model.fit_transform(data, label)

    vote = VotingClassifierMethod([1]*7)
    vote.fit(new_data, label)

    left = validate
    right1 = user_new
    right2 = question_new
    # left.columns = validate.columns
    right1_columns = [left.columns[1]]
    right1_columns.extend(list(range(right1.columns.size - 1)))
    right1.columns = right1_columns
    right2_columns = [left.columns[0]]
    right2_columns.extend(list(range(right2.columns.size - 1)))
    right2.columns = right2_columns
    left = pd.merge(left, right1, how='left', on=[validate.columns[1]])
    left = pd.merge(left, right2, how='left', on=[validate.columns[0]])
    data = left.drop(validate.columns, axis=1)
    label = left['label']
    from sklearn.preprocessing import StandardScaler, Imputer
    impute = Imputer()
    data = pd.DataFrame(impute.fit_transform(data), index=data.index)
    std = StandardScaler()
    data = pd.DataFrame(std.fit_transform(data), index=data.index)
    data_important = model.transform(data)
    p = vote.predict(data_important)
    df_pred = pd.concat([validate[['qid', 'uid']].ix[label.index, :],
                         pd.DataFrame(p[:, 1], columns=['label'])], axis=1)
    df_pred.to_csv('temp.csv', index=None, encoding='utf-8')
    print "temp.csv is generated with probabilities for bytecup validation set."
