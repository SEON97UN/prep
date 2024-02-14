import pandas as pd
import numpy as np

mpg = pd.read_csv("c:\\data_all/auto-mpg.csv", header=None)
mpg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
# print(mpg.info())


# print(mpg[['mpg']].mean())    # column이 하나여도 리스트로 만들어서 출력 -> Series가 아닌 DataFrame으로 출력됨
# print(mpg[['mpg', 'weight']].mean())

# print(mpg[['horsepower']].mean()) # 문자열 평균 구한 경우 - error

# 무의미한 컬럼의 기술통계가 같이 구해짐
mpg['origin'] = mpg['origin'].astype('str')
# print(mpg.describe())

# 상관계수와 공분산
# print(mpg[['mpg', 'cylinders', 'displacement']].corr())

# mpg 순으로 오름차순 정렬하고 동일한 값인 경우 displacement의 오름차순 정렬
# print(mpg.sort_values(by=['mpg','displacement'], ascending=[True, True]))

# 동일한 값은 순위의 평균
print(mpg.rank())

# 동일한 값은 낮은 순위를 부여
print(mpg.rank(method='min'))

