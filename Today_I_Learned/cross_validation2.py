# 2021.11.19
# 過学習について決定木モデルで検証

# 赤ワイン品質データセット

import pandas as pd
from sklearn import tree
from sklearn import metrics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

# df = pd.read_csv('winequality-red.csv')
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
df = pd.read_csv(url, sep=';')
print(df.head())

# 品質値対応表対応表(quality)
quality_mapping = {
    3:0,
    4:1,
    5:2,
    6:3,
    7:4,
    8:5
}

# pandasののmapを用いて与えられた辞書に基づき値を変換(目的変数)
df.loc[:, 'quality'] = df.quality.map(quality_mapping)


# 過学習

# frac=1 データフレームをシャッフル
df = df.sample(frac=1).reset_index(drop=True)

# データを上下に分割
# 上位1000
df_train = df.head(1000)

# 下位599
df_test = df.tail(599)


# 正答率を保存していくためのリストの初期化
# 学習用 検証用 初期値0.5
train_accuracies = [0.5]
test_accuracies = [0.5]

for depth in range(1, 25):
    # 決定木分類器の初期化
    clf = tree.DecisionTreeClassifier(max_depth=3)

    # 学習に利用する特徴量を指定
    cols = ['fixed acidity',
            'volatile acidity',
            'citric acid',
            'residual sugar',
            'chlorides',
            'free sulfur dioxide',
            'total sulfur dioxide',
            'density',
            'pH',
            'sulphates',
            'alcohol'
            ]

    # 与えられた特徴量と対応する目的変数でモデルを学習
    clf.fit(df_train[cols], df_train.quality)

    # 学習用データセットの予測
    train_predictions = clf.predict(df_train[cols])

    # 検証用データセットの予測
    test_predictions = clf.predict(df_test[cols])

    # 学習用データセットに対しての予測正答率
    train_accuracy = metrics.accuracy_score(df_train.quality, train_predictions)

    # 検証用データセットに対しての予測正答率
    test_accuracy = metrics.accuracy_score(df_test.quality, test_predictions)

    # リストに追加
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)


matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

plt.figure(figsize=(10,5))
sns.set_style('whitegrid')
plt.plot(train_accuracies, label='train accuracy')
plt.plot(test_accuracies, label='test accuracy')
plt.legend(loc='upper left', prop={'size': 15})
plt.xticks(range(0, 26, 5))
plt.xlabel('max_depth', size=20)
plt.ylabel('accuracy', size=20)
plt.show()
