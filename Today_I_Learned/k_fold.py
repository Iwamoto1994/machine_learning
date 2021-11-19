# 2021.11.19
# k-fold 交差検証
# stratified k-fold交差検証
# ホールドアウト検証
# leave-one-out 交差検証
# group k-fold 交差検証

import pandas as pd
from sklearn import model_selection

if __name__ == "__main__"

    # k-fold 交差検証 ----------------------------------------
    df = pd.read_csv('train.csv')

    df['kfold'] = -1 # 新しい列
    # ランダムサンプリング
    df = df.sample(frac=1)
    kf = model_selection.KFold(n_splits=5)

    for fold, (train_, valid_) in enumerate(kf.split(x=df)):
        df.loc[valid_, 'kfold'] = fold
    
    df.to_csv('train_folds.csv', index=False)


    # stratified k-fold ------------------------------------
    df = pd.read_csv('train.csv')

    df['kfold'] = -1
    # シャッフル
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for fold, (train_, valid_) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid_, 'kfold'] = fold
    
    df.to_csv('train_folds.csv', index=False)

    # memo
    # 各分割における目的変数の比率を一定に保てる
