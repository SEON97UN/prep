import pandas as pd
import numpy as np

# item.csv 파일을 읽어서 DataFrame 만들기
# item = pd.read_csv('c:\data_all/item.csv')
# # print(item.head())
# # item.info()

# # 현재 사용 중인 컬럼을 인덱스로 활용
# # item.index = item['code']
# # print(item)

# item.index = ['사과', '수박', '참외', '바나나', '레몬', '망고']
# print(item)

# 열 하나 선택
# print(item['price'])
# print(item.price) 
# 열을 선택할 때, list 이용 => DataFrame
# print(item[['price']])

#여러 열 선택
# print(item[['name', 'price']])

# # 행 선택 
# print(item.iloc[0]) # 0번째 행
# print(item.loc['사과']) # 사과라는 인덱스를 가진 행

# # 셀 선택
# print(item['name'][2]) # name 컬럼의 3번째 데이터

# print(item)
# print(item.iloc[1:4]) # 위치 인덱스에서는 마지막 위치 포함X
# print(item.loc["수박":"바나나"]) # 이름 인덱스에서는 마지막 위치 포함

# # price가 1500 미만인 행만 추출
# print(item[item['price']<1500])

# # price가 1000 ~ 1500
# print(item[(item['price']>=1000) & (item['price']<=1500)])

# # price 1000 또는 500
# print(item[item['price'].isin([1000,500])])

# 헤더가 없어서 컬럼 이름을 직접 설정
dt = pd.read_csv('c:\data_all/auto-mpg.csv', header=None)
dt.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'name']
# 처음 5개의 데이터만 확인
print(dt.head())

# 행과 열의 수 확인
print(dt.shape)
# 자료형 확인
print(dt.dtypes)
# 데이터 개수
print(dt.count())

# 앞의 3가지 정보 전부 확인 가능
# NULL(None)도 확인 가능
print(dt.info())

# 값의 개수와 빈도 수 확인
# NULL을 제외하고자 하는 경우 -> dropna=True
print(dt['cylinders'].value_counts())

# 
print(dt.describe())
print(dt.describe(include='all'))


print(dt['horsepower'].value_counts())