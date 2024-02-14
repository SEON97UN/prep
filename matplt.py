# line graph
import pandas as pd

# # header=0 : 첫 행이 컬럼이름
# mig_num = pd.read_excel("c:\\data_all/시도_별_이동자수.xlsx", header=0)
# # print(mig_num.head())

# # 엑셀에서 셀 병합이 있으면 첫 번째를 제외하고는 NaN으로 처리됨
# # NaN 데이터를 앞의 데이터로 채우기
# mig_num = mig_num.fillna(method='ffill')
# # print(mig_num.head())

# # 전출지별이 서울특별시 & 전입지별이 서울특별시 X 데이터 추출
# mask = (mig_num['전출지별'] == '서울특별시') & (mig_num['전입지별'] != '서울특별시')
# mig_seoul = mig_num[mask]
# # print(mig_seoul.head())

# # 전출지별 컬럼 제거
# mig_seoul.drop(['전출지별'], axis=1, inplace=True)
# # print(mig_num.head())

# # 전입지별 컬럼 이름 전입지로 변경
# mig_seoul.rename({"전입지별":"전입지"}, axis=1, inplace=True)
# # print(mig_seoul.head())

# # 전입지 인덱스로 설정
# mig_seoul.set_index('전입지', inplace=True)
# # print(mig_seoul.head())

# # 인덱스가 전라남도인 데이터를 추출
# sr_one = mig_seoul.loc['전라남도']
# # print(sr_one)


# line graph 그리기
import matplotlib.pyplot as plt
import numpy as np
# plt.plot(sr_one.index, sr_one.values)
# plt.show()


# 한글 처리

# 운영체제별 폰트 설정
from matplotlib import font_manager, rc
import platform
if platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == "Windows":
    font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
    rc('font', family=font_name)

# # 그래프 크기 설정 - 단위는 inch
# plt.figure(figsize=(14, 5))
# plt.xticks(size = 10, rotation = 'vertical')
# plt.title('서울 -> 전라남도', size = 30)

# # 막대 그래프 출력
# plt.hist(sr_one.index, sr_one.values, width=1.0)
# # 축 제목 설정 & 범례
# plt.xlabel('기간', size = 20)
# plt.ylabel('이동 인구수', size = 20)
# plt.legend(labels = ['서울 -> 전라남도'], loc = 'best', fontsize = 15)
# plt.show()
    
# 히스토그램
fruit = pd.read_csv('c:\\data_all/lovefruits.csv', encoding="cp949")
print(fruit)

# 선호과일 컬럼의 빈도수 추출
data = fruit['선호과일'].value_counts(sort=False)

'''
# 막대 그래프로 빈도수 출력 - 직접 빈도수 구해서 그려야 함
plt.bar(range(0, len(data), 1), data)
plt.xticks(range(0, len(data), 1), data.index)
plt.show()
'''

# histogram 그리기(hist 메서드가 직접 빈도수 구해서 그려줌)
plt.hist(fruit['선호과일'])
# plt.show()

# scatter plot
# 데이터 분포나 상관관계 파악
mpg = pd.read_csv("c:\\data_all/auto-mpg.csv", header=None)
mpg.columns = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin', 'name']

size = mpg['cylinders'] / mpg['cylinders'].max() *200
plt.scatter(x = mpg['weight'], y = mpg['mpg'], s = size, c = 'coral', alpha=.5)
plt.show()


