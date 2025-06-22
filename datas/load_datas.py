import pandas as pd

def load_data():
    train_df = pd.read_csv('./datas/train.csv')
    test_df = pd.read_csv('./datas/test.csv')

    data1 = train_df.copy(deep=True)
    data2 = test_df.copy(deep=True)

    return train_df, test_df, data1, data2
    