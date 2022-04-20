import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

TRAIN_FEATURES_PATH = '/content/drive/MyDrive/ml_task2/train_features.csv'
TEST_FEATURES_PATH = '/content/drive/MyDrive/ml_task2/test_features.csv'
TRAIN_LABELS_PATH = '/content/drive/MyDrive/ml_task2/train_labels.csv'

vitals = ['LABEL_RRate', 'LABEL_ABPm', 'LABEL_SpO2', 'LABEL_Heartrate']
tests = ['LABEL_BaseExcess', 'LABEL_Fibrinogen', 'LABEL_AST', 'LABEL_Alkalinephos', 'LABEL_Bilirubin_total',
         'LABEL_Lactate', 'LABEL_TroponinI', 'LABEL_SaO2',
         'LABEL_Bilirubin_direct', 'LABEL_EtCO2']
sepsis = ['LABEL_Sepsis']

def feature_engineering(df):

    #impute missing values
    df_ffill_bfill = df.groupby('pid').ffill().bfill()
    df_2 = df_ffill_bfill.fillna(df.mean())

    #construct the features
    df_age = df['Age'].groupby(by='pid', dropna=False).mean() #this is a bit hackerboy method lmao
    df_nan_count = df.drop(['Age'], axis=1).isnull().groupby(by='pid', dropna=False).sum()
    
    df_mean = df_2.drop(['Age'], axis=1).groupby(by='pid', dropna=False).mean()
    df_std = df_2.drop(['Age'], axis=1).groupby(by='pid', dropna=False).std()

    df_first = df_2.drop(['Age'], axis=1).groupby(by='pid', dropna=False).first()
    df_last = df_2.drop(['Age'], axis=1).groupby(by='pid', dropna=False).last()
    df_trend = df_last.subtract(df_first)
    
    df_features = pd.concat([df_age, df_nan_count.add_suffix('_nan_count'), df_mean.add_suffix('_mean'), df_std.add_suffix('_std'), df_trend.add_suffix('_trend'),
                             df_last.add_suffix('_last')], axis=1)
    return df_features

if __name__ == "__main__":
    #import the data
    data_features_train = pd.read_csv(TRAIN_FEATURES_PATH,index_col='pid').sort_values(by=['pid', 'Time'])
    data_features_test = pd.read_csv(TEST_FEATURES_PATH,index_col='pid').sort_values(by=['pid', 'Time'])
    data_labels_train = pd.read_csv(TRAIN_LABELS_PATH, index_col='pid').sort_values(by=['pid'])

    #impute data
    df = data_features_test.drop(['Time'], axis=1)
    df_ffill_bfill = df.groupby('pid').ffill().bfill()
    df_2 = df_ffill_bfill.fillna(df.mean())

    #create features
    df_nan_count = df.drop(['Age'], axis=1).isnull().groupby(by='pid', dropna=False).sum()

    X_12_train = feature_engineering(data_features_train.drop(['Time'], axis=1))
    X_12_test = feature_engineering(data_features_test.drop(['Time'], axis=1))

    X_3_train = feature_engineering(data_features_train.drop(['Time'], axis=1))
    X_3_test = feature_engineering(data_features_test.drop(['Time'], axis=1))

    #task 1&2
    clf = tests + sepsis
    predictions_12 = np.zeros((X_12_test.shape[0],len(clf)))

    for test_ in clf:
      y_train = data_labels_train[test_].to_numpy()
      pipe = Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(class_weight='balanced',criterion='entropy', n_estimators=800))])
      pipe.fit(X_12_train,y_train)
      y_pred = pipe.predict_proba(X_12_test)[:,1]
      predictions_12[:,clf.index(test_)] = y_pred

    df_12 = pd.DataFrame(predictions_12, index=data_features_test.index.unique(), columns=data_labels_train[clf].columns)

    #task 3
    predictions_3 = np.zeros((X_3_test.shape[0],len(vitals)))
    best_alphas = [460, 160, 380, 360]

    for vital_ in vitals:
        y_3_train = data_labels_train[vital_].to_numpy()
        model = Pipeline([('scaler', StandardScaler()), ('regr', Ridge(alpha=best_alphas[vitals.index(vital_)]))])
        model.fit(X_3_train, y_3_train)
        y_3_pred = model.predict(X_3_test)
        predictions_3[:,vitals.index(vital_)] = y_3_pred

    df_3 = pd.DataFrame(predictions_3, index=data_features_test.index.unique(), columns=data_labels_train[vitals].columns)

    #create submission and export
    submission = pd.concat((df_12, df_3), axis=1)
    submission.to_csv('prediction.zip', index=True, float_format='%.3f', compression='zip')

