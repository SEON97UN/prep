import pandas as pd
import numpy as np

# Data Preprocssing

# 자료형 변환
mpg = pd.read_csv('c:\\data_all/auto-mpg.csv', header=None)
mpg.columns=['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year','origin', 'name']

# 데이터가 많을 때 먼저 변경 - 예외가 발생하면 그 예외를 확인하면 이유를 알 수 있음
#result = mpg['horsepower'].astype('float')  
# 예외 발생 -> (1)치환 (2)제거
# (1)치환 - 주변값들과 비교

# ?를 NaN으로 치환
mpg['horsepower'].replace('?', np.nan, inplace=True)
# NaN 제거
mpg.dropna(subset=['horsepower'], axis=0, inplace=True)
# 실수 자료형으로 변경
mpg['horsepower'] = mpg['horsepower'].astype('float')

# 범주형 데이터를 의미를 갖는 문자열로 변환 - 출력이 목적
# replace에 dict 대입하면 key를 찾아서 value로 치환
mpg['origin'].replace({1:'USA', 2:'EU', 3:'JPN'}, inplace=True)
# print(mpg['origin'].unique())

# model year 컬럼의 자료형을 category로 변환
# 문자열이나 숫자 자료형은 원핫 인코딩이 안 되는 경우가 발생 가능
mpg['model year'] = mpg['model year'].astype('category')
# print(mpg.info())

student = pd.read_csv("c:\\data_all/student.csv", encoding='cp949', index_col='이름')
# print(student)


# 별다른 설정없이 그리면 한글이 출력되지 않음
import matplotlib.pyplot as plt
import platform
from matplotlib import font_manager, rc
if platform.system() == 'Darwin':
    rc('font', family = 'AppleGothic')
elif platform.system() == 'Windows':
    font_name = font_manager.FontProperties(
        fname = 'c:/Windows/Fonts/malgun.ttf'
    ).get_name()
    rc('font', family = font_name)
# 음수 출력 설정
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False

# 막대 그래프 그리기
student.plot(kind='bar')
# plt.show()

# 표준값과 편차값 구하기
kormean, korstd = student['국어'].mean(), student['국어'].std()
engmean, engstd = student['영어'].mean(), student['영어'].std()
mthmean, mthstd = student['수학'].mean(), student['수학'].std()

student['국어표준값'] = (student['국어'] - kormean) / korstd
student['영어표준값'] = (student['영어'] - engmean) / engstd
student['수학표준값'] = (student['수학'] - mthmean) / mthstd

# print(student)

# 편차값
student['국어편차값'] = student['국어표준값'] * 10 + 5
student['영어편차값'] = student['영어표준값'] * 10 + 5
student['수학편차값'] = student['수학표준값'] * 10 + 5

student[['국어편차값', '영어편차값', '수학편차값']].plot(kind='bar')
# plt.show()

from sklearn import preprocessing
# scickit-learn은 머신러닝을 위한 패키지라서 numpy 배열을 가지고 작업 수행
x = mpg[['horsepower']].values
'''
print('표준화 전의 기술 통계')
print("평균:", np.mean(x))
print("표준편차:", np.std(x))
print("최대:", np.max(x))
print("최소:", np.min(x))
'''
# 평균 = 0 & 표준편차 = 1 로 만드는 scaler
scaler = preprocessing.StandardScaler()
scaler.fit(x)
x_scaled = scaler.transform(x)
'''
print('표준화 후의 기술 통계')
print("평균:", np.mean(x_scaled))
print("표준편차:", np.std(x_scaled))
print("최대:", np.max(x_scaled))
print("최소:", np.min(x_scaled))
'''
# outlier가 많은 경우
# scaler = preprocessing.RobustScaler()

from sklearn.preprocessing import Normalizer

features = np.array([[1, 2], [2, 3], [4, 3]])

# 정규화 객체 생성

# l1
# 데이터 합 / 각 데이터를 나눈 값 : 다 더하면 1
normalizer1 = Normalizer(norm='l1')
result1 = normalizer1.transform(features)
# print(result1)

# l2
# 각 데이터 / 각 데이터 제곱의 합 sqrt
normalizer2 = Normalizer(norm='l2')
result2 = normalizer2.transform(features)
# print(result2)

# 
from sklearn.preprocessing import PolynomialFeatures
features = np.array([[2, 3], [3, 4], [1, 5]])
# 다항 및 교차항 생성해주는 객체 생성
polynomial_interaction = PolynomialFeatures(degree=2, include_bias = False)
result = polynomial_interaction.fit_transform(features)
# print(result)


# 특성 변환
# 함수를 적용해서 데이터 변경하는 것
from sklearn.preprocessing import FunctionTransformer

ftr = np.array([[1, 3], [3, 6]])
# 적용할 함수
def add_one(x:int) -> int:
    return x + 1
ones_transformer = FunctionTransformer(add_one)
rslt = ones_transformer.transform(ftr)
# print(rslt)


# 이산화 - 범주화
# 연속형 데이터를 간격 단위로 분할해서 값을 할당하는 작업

# 구간 생성
# 3개의 구간으로 분할
# 최소값부터 최대값까지 3개로 분할하고 데이터의 개수와 경계값을 리턴
count, bin_dividers = np.histogram(mpg['horsepower'], bins = 3)
# print(count) 
# print(bin_dividers)

# tuple
# 여러 개의 데이터를 묶어서 하나로 만들기 위한 자료형
# 테이블의 행을 만들기 위한 자료

# 생성
# (데이터, 데이터, ...)
# 괄호를 생략해도 튜플로 간주
# 10, 30 => (10, 30)

# 튜플은 한꺼번에 할당해도 되지만 나누어 할당 가능
# a, b = (100, 200)
# x = (100, 200)


# 각 그룹에 할당할 데이터 생성
bin_names = ['저출력', '보통출력', '고출력']

# 이산화
# x: 분할할 데이터, bins: 구간의 경계값 list, labels: 구간에 할당할 데이터 목록
# include_lowest: 첫 경계값 포함 여부
mpg['hp_bin'] = pd.cut(x = mpg['horsepower'], bins=bin_dividers, labels=bin_names, include_lowest=True)
# print(mpg[['horsepower', 'hp_bin']].head(20))

# numpy의 digitize를 이용한 구간 분할
age = np.array([[13], [30], [67], [36], [64], [24]])
# pandas의 메서드들은 DataFrame이나 Series를 이용해서 작업
# machine learning 관련된 메서드들은 2차원 이상의 ndarray를 가지고 작업 수행

# 30이 안 되면 0, 30 이상이면 1
# bins에는 여러 개의 데이터 설정이 가능
result = np.digitize(age, bins=[30])
# print(result)

# Binarizer를 이용한 구간 분할
from sklearn.preprocessing import Binarizer
# threshold(임계값)
# 흑백 이미지 데이터에서 뚜렷한 구분하기 위해서 임계값 설정
# 임계값 아래와 위를 구분
binarizer = Binarizer(threshold=30.0)
# print(binarizer.transform(age))

# KBinsDiscretizer를 이용한 구간 분할
from sklearn.preprocessing import KBinsDiscretizer
# 균등한 분포로 4분할
# ordinal: 라벨 인코딩 - 일련번호
kb_ordinal = KBinsDiscretizer(4, encode='ordinal', strategy='quantile')
# print(kb_ordinal.fit_transform(age))
# sparse
kb_sparse = KBinsDiscretizer(4, encode='onehot', strategy='quantile')
# print(kb_sparse.fit_transform(age))
# dense
kb_dense = KBinsDiscretizer(4, encode='onehot-dense', strategy='quantile')
# print(kb_dense.fit_transform(age))

from sklearn.cluster import KMeans
# sample data
sample = np.array([[13,30], [20,30], [21,99], [21,33], [98,22], [20,87]])
dataframe = pd.DataFrame(sample, columns = ['f1', 'f2'])
# print(dataframe)

clustering = KMeans(3, random_state=42)
clustering.fit(sample)
dataframe['group'] = clustering.predict(sample)
# print(dataframe)

# 직접 수식을 작성해서 탐지
features = np.array([[10, 10, 7, 6, 4, 5, 3, 3], [20000, 10, 7, 6, 4, 5, 3, 3]])
def outliers_z_score(ys):
    threshold = 3
    mean_y = np.mean(ys)
    stdev_y = np.std(ys)
    z_scores = [(y - mean_y)/stdev_y for y in ys]
    print("z_score", z_scores)
    return np.where(np.abs(z_scores) > threshold)
# print(outliers_z_score(features))

# 직접 수식을 작성해서 탐지 - 보정
# 데이터 개수에 관계없이 이상치 탐지 결과 출력됨
features = np.array([[10000, 10, 7, 6, 4, 5, 3, 3], [20000, 10, 7, 6, 4, 5, 3, 3]])
def outliers_mod_z_score(ys):
    threshold = 3.5
    median_y = np.median(ys)
    # 중위절대편차
    median_abs_deviation = np.median([np.abs(y-median_y) for y in ys])
    modified_z_scores = [.6745 * (y - median_y) / median_abs_deviation for y in ys]
    return np.where(np.abs(modified_z_scores) > threshold)
# print(outliers_z_score(features))

# IQR 이용하는 방법
# boxplot을 그렸을 때 수염 바깥쪽에 있는 데이터를 이상치로 간주

def outliers_iqr(ys):
    quantile_1, quantile_3 = np.percentile(ys, [25, 75])
    iqr = quantile_3 - quantile_1
    lower_bound = quantile_1 - (iqr * 1.5)
    upper_bound = quantile_3 + (iqr * 1.5)
    return np.where((ys > upper_bound) | (ys < lower_bound))
# print(outliers_iqr(features))

# print(outliers_iqr(mpg[['horsepower']].values))


# 일정한 비율을 이상치로 간주하는 방법

# 데이터를 생성해주는 API
from sklearn.datasets import make_blobs
# 일정한 비율로 이상치로 간주하는 API
from sklearn.covariance import EllipticEnvelope

# _는 데이터를 사용하지 않음
features, _ = make_blobs(n_samples=10, n_features=2, centers=1, random_state=42)
# 이상치 생성 - 0번 데이터는 이상한 데이터
features[0, 0] = 10000
features[0, 1] = 1000

print(features)
# 이상치 탐지 객체
outliers_detector = EllipticEnvelope(contamination = .1)
outliers_detector.fit(features)
# -1이면 이상치 1이면 정상적인 데이터
outliers_detector.predict(features)


# 연습문제
# 파일은 공백으로 구분
# 첫번째 데이터 - 접속한 컴퓨터의 IP
# 마지막 데이터 - 트래픽
# 전체 트래픽의 합계 구하기
# IP별 트래픽의 합계
# 방법 - 코테 -> numpy or pandas 사용 X 
#      - 분석 -> numpy or pandas 사용
import pandas as pd
import numpy as np
data = pd.read_table('c:\\data_all/log.txt', header=None)
# print(data.head()

ips = []
traffics = []

for line in data[0]:
    parts = line.split()
    ips.append(parts[0])
    traffics.append(parts[-1])

df = pd.DataFrame({
    'IP': ips,
    'Traffic': traffics
})

# print(df.info())

# (1)전체 트래픽의 합계 구하기
df['Traffic'].replace('-', np.nan, inplace=True)
df['Traffic'].replace('"-"', np.nan, inplace=True)
df.dropna(subset=['Traffic'], axis=0, inplace=True)
try:
    df['Traffic'] = df['Traffic'].astype('int')
except:
    print('cannot convert')

total_Traffic = df['Traffic'].sum()
print('전체 트래픽의 합계:', total_Traffic)

# (2) IP별 트래픽의 합계
total_Traffic_by_IP = df.groupby('IP')['Traffic'].sum()
print('IP별 트래픽의 합계:', total_Traffic_by_IP)


# List Comprehension
l = list(range(10))

# l의 모든 데이터를 순회하면서 각 데이터의 제곱을 한 값을 가지고 새로운 list 생성
# (1) for문 & .append()
r = []
for i in l:
    r.append(i ** 2)
# print(r)
# (2) map 함수 이용
r = list(map(lambda x : x ** 2, l))
# print(r)
# (3) List Comprehension
r = [i ** 2 for i in l]
# print(r)

# filtering  - l의 모든 데이터를 i에 순차적으로 대입해서 뒤의 조건문이 참인 경우만 list 생성
r = [i for i in l if i % 2 == 0]
# print(r)

# for 안에 for 사용
# 먼저 나온 for가 바깥쪽 for이고 뒤에 나온 for가 안쪽 for
li1 = [1,2,3]
li2 = [4,5,6]
r = [x * y for x in li1 for y in li2]
# print(r)

r = [x for x in li1 for y in li2]
# print(r)