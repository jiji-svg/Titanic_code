import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def visualing_columns(data1):
    plt.figure(figsize=(16,12))
    sns.countplot(data=data1, x='Sex', hue='Survived')
    plt.title('생존 여부에 따른 성별 분포')

    plt.figure(figsize=(16,12))
    sns.countplot(data=data1, x='Pclass', hue='Survived')
    plt.title('객실 등급에 따른 생존율')

    fig, saxis = plt.subplots(nrows=5, figsize=(16,16))

    sns.barplot(x='Sex', y='Survived', data=data1, ax=saxis[0])
    saxis[0].set_title('성별과 생존율')

    sns.barplot(x='Pclass', y='Survived', data=data1, ax=saxis[1])
    saxis[1].set_title('객실 등급과 생존율')

    sns.histplot(x='Age', hue='Survived', data=data1,
                bins=30, kde=True, ax=saxis[2])
    saxis[2].set_title('나이와 생존율')

    sns.barplot(x='FareBand', y='Survived', data=data1, ax=saxis[3])
    saxis[3].set_title('요금과 생존율')

    sns.countplot(x='Embarked', hue='Survived', data=data1, ax=saxis[4])
    saxis[4].set_title('탑승 항구와 생존율')

    plt.tight_layout()
    plt.show()

def visualize_features(X,y, model):
    model.fit(X, y)
    feat_imp = pd.Series(model.feature_importances_, index=X.columns)
    feat_imp.sort_values().plot(kind='barh')
    plt.title('feature importance')
    plt.show()
