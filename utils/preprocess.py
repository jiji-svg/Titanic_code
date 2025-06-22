import pandas as pd

title_mapping = {'Mr': 1, 'Miss':2, 'Mrs':3, 'Master':4, 'Rare':5}

def is_null(data_cleaner):
    for dataset in data_cleaner:
        dataset['Sex'] = dataset['Sex'].map({'male':0, 'female':1})
        dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
        dataset['Embarked'].fillna(dataset['Embarked'].mode()[0], inplace=True)
        dataset['Embarked'] = dataset['Embarked'].map({'S':0, 'C':1, 'Q':2})
        dataset['Fare'].fillna(dataset['Fare'].mean(), inplace=True)
        dataset['FareBand'] = pd.qcut(dataset['Fare'], 4, labels=False)
        dataset['AgeGroup'] = pd.cut(dataset['Age'],
                                bins=[0,12,18,35,60,100],
                                labels=False)
        dataset['FareBand'] = pd.qcut(dataset['Fare'],
                                  4,
                                  labels=False)
        
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize']==1, 'IsAlone'] = 1
        dataset['Title'] = dataset['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 
                                                    'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)
        dataset['HasCabin'] = dataset['Cabin'].notnull().astype(int)
        dataset['Deck'] = dataset['Cabin'].str[0]
        dataset['Deck'] = dataset['Deck'].fillna('U')
        deck_mapping = {'A':1, 'B':2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7,
                    'T':8, 'U':0}
        dataset['Deck'] = dataset['Deck'].map(deck_mapping)

    