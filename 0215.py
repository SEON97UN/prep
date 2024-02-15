# 중복 데이터 제거
# 중복 데이터 존재 -> 분석 결과 왜곡 가능

import pandas as pd
import numpy as np

df = pd.DataFrame([['안녕하세요', '안녕하세요', '헬로우', '키아 오라', '반갑습니다'], ['한국', '한국', '미국', '뉴질랜드', '한국']])
df = df.T
df.columns = ['인사말', '국가']
# print(df)

# 데이터 확인
# 중복 확인
# print(df.duplicated()) #모든 컬럼의 값이 중복된 경우 확인
# print(df.duplicated(subset=['국가'])) #특정 컬럼(국가)의 값이 중복된 경우 확인

# 중복 제거
# print(df.drop_duplicates(subset=['국가'])) #default는 앞의 데이터 보존
# print(df.drop_duplicates(subset=['국가'], keep='last')) # keep 옵션 -> 마지막 데이터 보존

# Titanic Data
import seaborn as sns
titanic = sns.load_dataset('titanic')
# titanic.info()

# apply 함수 적용
df_t = titanic[['sex', 'age']]
# print(df_t.head())

# 실수 1개 받아서 1을 더해서 리턴하는 함수 
def plusone(data:float) -> float:
    return data + 1
# 함수 만들어서 적용
df_t1 = df_t['age'].apply(plusone)
# print(df_t1)

# 람다 함수를 이용해서 위와 동일한 작업을 수행
df_t2 = df_t['age'].apply(lambda x : x+1)
# print(df_t2)

# print(df_t2 == df_t1)

# sex 열의 모든 내용을 대문자로 변경 - 문자열 클래스에서 대문자로 변경해주는 메서드가 있는지 확인
df_t3 = df_t['sex'].apply(str.upper)
# print(df_t3.head())

# Series를 받아서 각 데이터가 null인지 확인해주는 함수
'''
def missing_value(series: pd.Series) -> pd.Series:
    return series.isnull()
result = df_t.apply(missing_value)
'''
# print(result.head())
# print(sum((result['sex'] == True) & sum(result['age'] == True)))


# pipe 
# Series 받아서 NaN 여부 리턴하는 함수
def missing_value(x: pd.Series) -> pd.Series:
    return x.isnull()

# Series 받아서 True 개수 리턴하는 함수
def missing_count(x: pd.Series) -> int:
    return missing_value(x).sum()

# DataFrame 받아서 총 NaN 개수 리턴하는 함수
def total_number_missing(x: pd.DataFrame) -> int:
    return missing_count(x).sum()

tisub = titanic[['age', 'embarked']]
# tisub.info()

# Series 받아서 Series 리턴하는 함수
# DataFrame 리턴
# print(tisub.pipe(missing_value))

# Series 받아서 1개의 데이터(정수) 리턴하는 함수 적용
# Series 리턴 - 각 열의 결과
# print(tisub.pipe(missing_count))

# DataFrame 받아서 집계 한 후 하나의 값을 리턴하는 함수 적용
# 하나의 값 리턴
# print(tisub.pipe(total_number_missing))


# 열 이름 전부 가져오기
columns = list(titanic.columns.values)
# print(type(columns))
# print(columns)

# 컬럼 이름을 알파벳 순으로 정렬해서 재배치
df_srt = titanic[sorted(columns)]
# print(df_srt)


# 엑셀 데이터 불러오기
dt = pd.read_excel('c:\\data_all/주가데이터.xlsx')
# dt.info()

# 공백을 기준으로 분할 - list 리턴
message = "Hello Universe"
ar = message.split(" ")
# print(ar)

# 날짜 컬럼을 문자열로 변환
dt['연월일'] = dt['연월일'].astype('str')
# - 를 기준으로 분할
dates = dt['연월일'].str.split("-")
# print(dates.head())
dt['연'] = dates.str.get(0)
dt['월'] = dates.str.get(1)
dt['일'] = dates.str.get(2)
# print(dt.head())

titanic = pd.read_csv("c:\\data_all/titanic.csv")
# age가 10-19 인 데이터만 추출
condition = (titanic['age'] >= 10) & (titanic['age'] <= 19)
result = titanic.loc[condition, :]
# print(result.head())
# print(result['age'].unique()) 

# age가 10 미만 & sex가 female인 데이터 중에서 age, sex, alone, 컬럼만 추출
cd = (titanic['age'] < 10) & (titanic['sex'] == 'female')
rslt = titanic.loc[cd, ['age', 'sex', 'alone']]
# print(rslt.head())

# sibsp 가 3, 4, 5 인 데이터 추출
# .isin()
cd1 = (titanic['sibsp'].isin([3, 4, 5]))
rslt1 = titanic.loc[cd1, : ]
# print(rslt1['sibsp'].unique())


# concat
df1 = pd.DataFrame({'a':['a0', 'a1', 'a2', 'a3'],
                    'b':['b0', 'b1', 'b2', 'b3'],
                    'c':['c0', 'c1', 'c2', 'c3']},
                    index = [0, 1, 2, 3])
# print(df1)
df2 = pd.DataFrame({'a':['a4', 'a5', 'a6', 'a7'],
                    'b':['b4', 'b5', 'b6', 'b7'],
                    'd':['d4', 'd5', 'd6', 'd7']},
                    index = [2, 3, 4, 5])
# concat - default -> Set 연산처럼 세로 방향으로 합쳐짐
# 컬럼의 이름이 같은 경우는 바로 합쳐지지만, 다른 경우는 반대편의 NaN을 갖는 컬럼을 생성해서 합쳐줌
# print(pd.concat([df1, df2]))

# 좌우로 합치기 - axis = 1
# 다른 옵션이 없으면 index를 기준으로 합쳐짐
# Outer Join 처럼 수행됨
# print(pd.concat([df1, df2], axis = 1))

# Inner Join
# print(pd.concat([df1, df2], axis = 1, join='inner'))

# append
# print(df1.append(df2))
# print(df1._append(df2))

# combine_first
# print(df1.combine_first(df2))

stock_p = pd.read_excel('c:\\data_all/stock price.xlsx')
stock_v = pd.read_excel('c:\\data_all/stock valuation.xlsx')
# print(stock_p.head())
# print(stock_v.head())

# 아무 옵션X -> 동일한 이름의 컬럼 가지고 JOIN
# INNER JOIN
# print(pd.merge(stock_p, stock_v))

# FULL OUTER JOIN
# print(pd.merge(stock_p, stock_v, how='outer', on='id'))

# JOIN 하는 컬럼의 이름이 다를 때
# print(pd.merge(stock_p, stock_v, how='right', left_on='stock_name', right_on= 'name'))

# JOIN은 기본적으로 index를 가지고 JOIN을 수행
stock_p.index = stock_p['id']
stock_v.index = stock_v['id']
# 동일한 컬럼 이름 제거
stock_p.drop(['id'], axis=1, inplace=True)
stock_v.drop(['id'], axis=1, inplace=True)
# print(stock_p.join(stock_v))


# group by
tit = titanic[['age', 'sex', 'class', 'fare', 'survived']]
# print(tit.head())

# 그룹화
grouped = tit.groupby(['class'])
# 그룹화 한 후 각 그룹의 데이터 개수 출력
for key, group in grouped:
    print(key, len(group))

# 특정 그룹의 데이터 선택
group3 = grouped.get_group('Third')
# print(group3.head())

# 2개의 열로 그룹화 - key가 2개 항목의 tuple로 만들어짐
grouped = tit.groupby(['class', 'sex'])
for key, group in grouped:
    print(key, len(group))

group3m = grouped.get_group(('Third', 'male'))
# print(group3m)

# 제공되는 집계함수를 호출
tit = titanic[['class', 'age']]
grouped = tit.groupby(['class'])
std_all = grouped.std()
print(std_all)

# 사용자 정의 함수를 적용하고자 할 때는 agg 함수 호출
def min_max(x):
    return x.max() - x.min()
result = grouped.agg(min_max)
# print(result)

# age 열의 값을 z-score로 변환
# z-score: (값-vudrbs) / 표준편차

def z_score(x):
    return(x-x.mean()) / x.std()
age_zsc = grouped.transform(z_score)
# print(age_zsc)

dfn = titanic[['class', 'sex', 'age']]
# 2개의 컬럼을 이용해서 그룹화
grouped = dfn.groupby(['class', 'sex'])
# print(grouped)
for key in grouped:
    print(key)

gdf = grouped.mean()
# print(gdf)

# 특정 class 출력
# print(gdf.loc['First'])
# 특정 sex 출력
# print(gdf.xs('male', level = 'sex'))
# 특정 행 출력
# print(gdf.loc['First', 'female'])


# pivot_table
# help(pd.pivot_table)
pv_rslt = pd.pivot_table(dfn, values='age', index='class', columns='sex', aggfunc='sum')
# print(pv_rslt)

# stacked
mul_index = pd.MultiIndex.from_tuples([('cust_1', '2015'), ('cust_1', '2016'), ('cust_2', '2015'), ('cust_2', '2016')])
data = pd.DataFrame(data = np.arange(16).reshape(4,4), index = mul_index, columns = ['prd_1', 'prd_2', 'prd_3', 'prd_4'], dtype = 'int')
print(data)
stacked = data.stack()
print(stacked)

unstacked = stacked.unstack(level=0)
# print(unstacked)
# print(unstacked.unstack())

# 숫자 데이터의 전처리
# 단위 환산
# mpg 데이터는 자동차에 대한 정보를 가지는 데이터
# mpg 열이 연비인데 미국에서는 갤런당 마일 사용 <-> 우리나라는 리터당 킬로미터
# 1mile = 1.60934km
# 1gallon = 3.78541L
mpg = pd.read_csv('c:\\data_all/auto-mpg.csv', header=None)
mpg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']
# print(mpg.head())

# gallon per mile to liter per kilometer
mpg_to_kpi = 1.60934 / 3.78541
mpg['kpl'] = mpg['mpg'] * mpg_to_kpi
# print(mpg.head())

# print(mpg.dtypes)

# ? 데이터 존재 -> 형 변환 실패

print(mpg['horsepower'].unique()) # 중복된 데이터 제거하고 모든 데이터 추출
# ?를 NaN으로 치환
mpg['horsepower'].replace('?', np.nan, inplace=True)
# NaN 제거
mpg.dropna(subset=['horsepower'], inplace=True)
# 실수 자료형으로 변경
mpg['horsepower'] = mpg['horsepower'].astype('float')

# print(mpg.dtypes)