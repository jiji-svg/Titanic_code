import joblib
import pandas as pd
from datas import load_data
from utils import is_null

def test():
    train_df, test_df, data1, data2 = load_data()
    data_cleaner = [data1, data2]
    is_null(data_cleaner)

    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    data2 = data2.drop(drop_cols, axis=1)

    # 저장된 파이프라인 불러오기 (훈련된 전처리 + 모델 포함)
    pipeline = joblib.load('saved/saved_models/voting_pipeline.pkl')

    # 예측
    preds = pipeline.predict(data2)

    # PassengerId가 data2에서 drop되었기 때문에 원본 test_df에서 가져와야 함
    submission = pd.DataFrame({
        'PassengerId': test_df['PassengerId'],
        'Survived': preds
    })

    submission.to_csv('saved/saved_submissions/submission1.csv', index=False)
    print("submission.csv 파일이 생성되었습니다.")

if __name__ == '__main__':
    test()