import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# pip install pandas-datareader
# ライブラリの読み込み
import pandas_datareader as web



# N225
stock_code = '^NKX'
data_source = 'stooq'
start = '1986-09-03'
end = '2021-11-22'

# 株情報を取得する
df = web.DataReader(stock_code, data_source=data_source, start=start, end=end)

print(df)
df.reset_index()



sorted_df = df.sort_values(['Date'], ascending=True)
N225 = sorted_df.reset_index()
# N225.head()

# 一旦保存
N225.to_csv("N225.csv")


N225['Close_LN'] = np.log(N225['Close'])
N225['Close_LN_diff'] = N225['Close_LN'].diff()
N225['Close_LN_diff_100'] = N225['Close_LN_diff'] * 100

threshold = np.log(1.01) * 100


# r >= +1% (1%以上)
def one_hot_r1(r):
    if r >= threshold:
        return 1
    else:
        return 0

# +1% > r >= 0% (0%以上)
def one_hot_r1_0(r):
    if r < threshold and r >= 0:
        return 1
    else:
        return 0

# 0% > r >= -1% (-1%以上) 
def one_hot_r0_m1(r):
    if r < 0 and r >= -(threshold):
        return 1
    else:
        return 0

# -1% > r (-1%未満)
def one_hot_rm1(r):
    if r < -(threshold):
        return 1
    else:
        return 0



N225['r>=1%_tmp'] = N225['Close_LN_diff_100'].apply(one_hot_r1)
N225['r>=0%_tmp'] = N225['Close_LN_diff_100'].apply(one_hot_r1_0)
N225['r>=-1%_tmp'] = N225['Close_LN_diff_100'].apply(one_hot_r0_m1)
N225['-1%>r_tmp'] = N225['Close_LN_diff_100'].apply(one_hot_rm1)

# N225.head()

N225['r>=1%'] = N225['r>=1%_tmp'].shift(-1)
N225['r>=0%'] = N225['r>=0%_tmp'].shift(-1)
N225['r>=-1%'] = N225['r>=-1%_tmp'].shift(-1)
N225['-1%>r'] = N225['-1%>r_tmp'].shift(-1)

# N225.head()
# N225.tail()



N225_ = N225[1:]
N225_ = N225_[:-1]


# SettingWithCopyWarning: 
N225_['r>=1%'] = N225_['r>=1%'].astype(int)
N225_['r>=0%'] = N225_['r>=0%'].astype(int)
N225_['r>=-1%'] = N225_['r>=-1%'].astype(int)
N225_['-1%>r'] = N225_['-1%>r'].astype(int)

N225_.to_csv("N225_2.csv")

