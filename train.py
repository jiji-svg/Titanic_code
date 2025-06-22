import joblib
from datas import load_data
from utils import visualing_columns, is_null
from models import VoteModel
from utils import fonting

def main():
    train_df, test_df, data1, data2= load_data()
    data_cleaner = [data1, data2]
    is_null(data_cleaner)
    # fonting()
    # visualing_columns(data1)
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data1 = data1.drop(drop_cols,axis=1)
    data2 = data2.drop(drop_cols, axis=1)
    y = data1['Survived']
    X = data1.drop('Survived', axis=1)
    model = VoteModel(X, y)
    acc = model.train()
    print(acc)
    joblib.dump(model.final_pipeline, 'saved/saved_models/voting_pipeline.pkl')


if __name__ == '__main__':
    main()