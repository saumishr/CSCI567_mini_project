import pandas as pd
from keras.preprocessing import sequence
import os
from utils import *

DIR = 'data'
USER = os.path.join(DIR, 'user_info.txt')
QUESTION = os.path.join(DIR, 'question_info.txt')
TRAIN = os.path.join(DIR, 'invited_info_train.txt')
VALID = os.path.join(DIR, 'validate_nolabel.txt')
TEST = os.path.join(DIR, 'test_nolabel.txt')

QLABEL_NUM = 20
ULABEL_NUM = 144
QWORDSEQ_MAXLEN = 17
UWORDSEQ_MAXLEN = 47
QWORDSEQ_SIZE = 13232
UWORDSEQ_SIZE = 31100


def get_question_user():
    question = pd.read_csv(QUESTION, delimiter='\t', header=None)
    question.columns = ('qid', 'qlabel', 'qwordseq', 'qcharseq', 'likenum', 'ansnum', 'bansnum')
    qlabel = np.zeros((len(question), QLABEL_NUM))

    user = pd.read_csv(USER, delimiter='\t', header=None)
    user.columns = ('uid', 'ulabel', 'uwordseq', 'ucharseq')
    ulabel = np.zeros((len(user), ULABEL_NUM))

    for i, label in enumerate(question.qlabel):
        if label == '/':
            continue
        qlabel[i, int(label)] = 1
    qlabel = pd.DataFrame(qlabel, columns=('qlabel_{}'.format(i) for i in range(QLABEL_NUM)))

    for i, label in enumerate(user.ulabel):
        if label == '/':
            continue
        label = [int(x) for x in label.split('/')]
        ulabel[i, label] = 1
    ulabel = pd.DataFrame(ulabel, columns=('ulabel_{}'.format(i) for i in range(ULABEL_NUM)))

    question = pd.concat([question.drop(['qlabel', 'qcharseq'], axis=1), qlabel], axis=1)
    user = pd.concat([user[['uid', 'uwordseq']], ulabel], axis=1)

    return question, user

def load_dateset(question, user):
    train = pd.read_csv(TRAIN, header=None, delimiter='\t')
    train.columns = ('qid', 'uid', 'label')
    train_x = train[['qid', 'uid']]
    train_y = train['label']
    train_x = pd.merge(train_x, question, on='qid', how='left')
    train_x = pd.merge(train_x, user, on='uid', how='left')
    train_x = train_x.drop(['qid', 'uid'], axis=1)

    test = pd.read_csv(VALID)
    test_x = test[['qid', 'uid']]
    test_x = pd.merge(test_x, question, on='qid', how='left')
    test_x = pd.merge(test_x, user, on='uid', how='left')
    test_x = test_x.drop(['qid', 'uid'], axis=1)
    return (train_x, train_y), (test_x,)

def train_model(train, test):
    def transform(words):
        dic = {}
        result = []
        for word in words:
            a = []
            if word != '/':
                for idx in word.split('/'):
                    if idx not in dic:
                        dic[idx] = len(dic) + 1
                    a.append(dic[idx])
            result.append(a)
        return result

    train_x, train_y = train
    test_x = test[0]

    train_qwordseq = sequence.pad_sequences(transform(train_x['qwordseq']), maxlen=QWORDSEQ_MAXLEN)
    train_uwordseq = sequence.pad_sequences(transform(train_x['uwordseq']), maxlen=UWORDSEQ_MAXLEN)
    test_qwordseq = sequence.pad_sequences(transform(test_x['qwordseq']), maxlen=QWORDSEQ_MAXLEN)
    test_uwordseq = sequence.pad_sequences(transform(test_x['uwordseq']), maxlen=UWORDSEQ_MAXLEN)
    train_x = train_x.drop(['qwordseq', 'uwordseq'], axis=1).values
    test_x = test_x.drop(['qwordseq', 'uwordseq'], axis=1).values
    train_y = train_y.values

    def lambda_output_shape(input_shape):
        return (None, sum(shape[1] for shape in input_shape))

    features = np.concatenate((train_qwordseq, train_uwordseq), axis=1)
    features = np.concatenate((train_x, features), axis=1)
    where_are_NaNs = np.isnan(features)
    features[where_are_NaNs] = 0
    test_features = np.concatenate((test_qwordseq, test_uwordseq), axis=1)
    test_features = np.concatenate((test_x, test_features), axis=1)
    where_are_NaNs = np.isnan(test_features)
    test_features[where_are_NaNs] = 0

    archs = [[features.shape[1], 800, 500, 300, 1]]
    reg_coeffs = [0.0]

    model = testmodels(features, train_y, None, None, archs, 'relu', 'softmax', reg_coeffs)

    predictions = model.predict_proba(test_features, verbose=1)

    writer = open("op.txt", "w")
    for value in predictions:
        writer.write(str(value) + "\n" )
    writer.close()

def main():
    question, user = get_question_user()
    train, test = load_dateset(question, user)
    train_model(train, test)

    writer = open("temp.csv", "w")
    reader1 = open(VALID, "r")
    l1 = reader1.readline()
    writer.write(l1)
    l1 = reader1.readline()
    reader2 = open("op.txt", "r")
    m1 = reader2.readline()
    while l1 != "":
        newline = l1.strip() + "," + m1[1:-2].strip()
        writer.write(newline + "\n")
        l1 = reader1.readline()
        m1 = reader2.readline()

    writer.close()
    reader1.close()
    reader2.close()


if __name__ == '__main__':
    main()
