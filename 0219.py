import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 이상치 처리
houses = pd.DataFrame()
houses['price'] = [534433, 392333, 293222, 4322032]
houses['bedroom'] = [2, 3.5, 2, 116]
houses['square_feet'] = [1500, 2500, 1500, 48000]
# print(houses)

# bedroom이 20개 이상 여부를 별도의 특성으로 생성
# 이상한 데이터를 무시할 수 없는 경우 별도의 특성으로 생성하기도 함
houses['outlier'] = np.where(houses['bedroom'] < 20, 0, 1) #20보다 작으면 0, 크면 1
# print(houses)

# 값의 크기를 줄이기 - 표준화
# 값의 편차가 큰 데이터의 영향력이 줄어들게됨
# Use Scaler of scikit-learn  &  Log Scaling
houses['Log_Of_Square_Feet'] = [np.log(x)for x in houses['square_feet']]
# print(houses)

from sklearn import preprocessing
# RobustSclaer 
df = pd.DataFrame(houses['bedroom'])
# Scaling: transform into -1 ~ 1 or 0 ~ 1 
scaler = preprocessing.RobustScaler()
scaler.fit(df)
x_scaled = scaler.transform(df)
houses['scaled_bedroom'] = x_scaled
# print(houses)

# Check out Missing Values with Titanic Data
import seaborn as sns
titanic = sns.load_dataset('titanic')
titanic.info() # don't have to use print when using .info()

# deck - isnull
# print('Sum of Missing Values:', titanic['deck'].isnull().sum()) #sum of missing values: 688

# deck - value_counts
# Using value_counts can calculate each columns's missing values
# print(titanic['deck'].value_counts(dropna=False)) 

# Handling Missing Values
# if the number of Missing Values is more than 500, then dropna
result = titanic.dropna(axis=1, thresh=500)
# print(result.info())  #deck column has been dropped

# 특정 열의 값이 NaN -> 행 삭제
result = result.dropna(subset=['age'], axis = 0)
# print(result.info())


# 결측값 대체 - 앞의 데이터(828행의 데이터)로 대체
# print(titanic['embark_town'][825:831]) #829행에 NaN
result = titanic['embark_town'].fillna(method='ffill')
# print(result[825:831])

# substitute mode for NaN
counts = titanic['embark_town'].value_counts()
# print(counts)
mode = counts.idxmax() #최빈 인덱스
# print(mode)
result = titanic['embark_town'].fillna(mode)
print(result[825:831]) #829행 데이터가 mode로 채워짐

# 머신러닝 알고리즘을 이용해 채우기
# pip install fancyimpute
'''
from fancyimpute import KNN

features = np.array([[200, 250], [100, 200], [300, 600], [400, 290], [500, 380], [110, np.nan]])
result = KNN(k=5).fit_transform(features)
print(result)
'''
# one hot encoding
result = pd.get_dummies(titanic['class'])
# print(result.head(10))

# scikit-learn을 이용한 one hot encoding
# pandas의 get_dummies 와 동일한 작업을 수행
one_hot = preprocessing.LabelBinarizer()
result = one_hot.fit_transform(titanic['class'])
# print(result)
# 원래 데이터로 복원
print(one_hot.inverse_transform(result))


# 멀티 클래스 원핫 인코딩
# 컬럼의 개수가 일치해야 함
# 문자열을 가지고 거리 계산을 할 때, 가장 먼저 하는 일 중 하나가 컬럼의 개수를 맞추는 것
m_features = [('Java', 'C++'), ('C++', 'Python'), ('C#', 'JavaScript'), ('Java', 'R'), 
              ('Python', 'Scala'), ('Golang', 'Pyhton')]
one_hot_multi = preprocessing.MultiLabelBinarizer()
# 여러 컬럼에 걸쳐 나오는 모든 경우를 각각의 컬럼으로 생성 -> one hot encoding
result = one_hot_multi.fit_transform(m_features)
# print(result)

# 각각의 컬럼이 의미하는 바 - classes or categories
# print(one_hot_multi.classes_)

# 일련 번호 형태로 encoding
encoder = preprocessing.LabelBinarizer()
result = encoder.fit_transform(titanic['class'])
# print(result)
 
# 특정한 순서대로 인코딩
df = pd.DataFrame({'score': ['저조', '보통', '우수']})
encoder = preprocessing.LabelBinarizer()
result = encoder.fit_transform(df['score'])
# print(result)
# 저조에 0, 보통에 1, 우수는 2로 할당 -> replace로 치환
mapper = {'저조': 0, '보통':1, '우수':2}
result = df['score'].replace(mapper)
# print(result)

dt = '아침'
mapper = {'아침': 1, '점심':2, '저녁':3}
# print(mapper[dt])

# OrdinalEncoding
# 데이터를 정렬해서 인코딩 - 숫자도 문자열로 변경해서 정렬
# 대소문자 구별하지 않고 인코딩 되도록 변경 - Think about it
features = np.array([['Low', 10], ['High', 30], ['Medium', 20]]) # 숫자도 문자열로 변경해서 정렬
encoder = preprocessing.OrdinalEncoder()
# print(encoder.fit_transform(features))


# 누락된 데이터 대체 - KNN 이용
from sklearn.neighbors import KNeighborsClassifier

# 첫번째 열이 범주 & 나머지 2개는 숫자 데이터
X = np.array([[0, 2.10, 1.45], [1, 1.10, .45], [0, 2.38, .97], [1, .10, .65]])

# 누락된 데이터
# 데이터의 값을 예측해서 fill
X_with_nan = np.array([[np.nan, 2.0, 1.1], [np.nan, .98, .55]])

# 모델 생성 - 근방 3개 찾아 예측
clf = KNeighborsClassifier(3, weights='distance')

# 2개의 feature로 target 예측
trained_model = clf.fit(X[: , 1:], X[:, 0])
imputed_values = trained_model.predict(X_with_nan[:, 1:])

# print(imputed_values)

# 예측된 값을 본 데이터에 합침
# numpy는 stack을 사용 // pandas는 merge나 join 사용

# 머신러닝 결과로 나온 예측값과 feature를 좌우로 결합
# 머신러닝의 데이터 or 결과는 기본이 2차원 배열
# reshape는 차원을 변경하는 함수
X_with_impute = np.hstack((imputed_values.reshape(-1, 1), X_with_nan[:, 1:]))
# print(X_with_impute)

# 결측치가 없는 데이터와 결측치를 채운 데이터 상하 결합
# print(np.vstack((X, X_with_impute)))

# 분류기에 가중치 부여 - class_weight
from sklearn.ensemble import RandomForestClassifier
list1 = []
for i in range(0, 10, 1):
    list1.append(0)
list2 = []
for i in range(0, 90, 1):
    list2.append(1)
target = np.array(list1+list2)
print(target)

# 샘플 개수 확인
iclass0 = np.where(target == 0)[0]
iclass1 = np.where(target == 1)[0]
print(iclass0)
print(iclass1)

# 시드 고정
np.random.seed(42)

# 다운 샘플링 - 비복원 추출
iclass1_downsample = np.random.choice(iclass1, size = len(iclass0), replace=False)
# print(iclass1_downsample)

# 업 샘플링 - 복원 추출
# 단순한 복제는 잘 사용 X
iclass0_upsample = np.random.choice(iclass0, size=len(iclass1), replace=True)
# print(iclass0_upsample)


# 문자열 클래스의 메서드를 이용한 정제
text_data = [' Hello Python', 'And Sorry I could not   ', 'And be one, long']
# (1) 좌우 공백 제거
result = [string.strip() for string in text_data]
# print(result)
# (2) , 제거
result = [string.replace(",", "") for string in result]
# print(result)
# (3) 모두 대문자로 변경
result = [string.upper() for string in result]
# print(result)

import re
match = re.match('[0-9]', '안녕하세요. 반갑습니다.')
print(match)
match = re.match('[0-9]', ' 125698')
print(match)
match = re.match('[0-9]+', '12346849')
print(match)
# 공백문자로 시작하고 다음에 숫자 등장
match = re.match('\s[0-9]', ' 1234')
print(match)

string = '''
기상청은 슈퍼컴퓨터도 서울지역의 집중호우를 제대로 예측하지 못했다고 설명했습니다.
왜 오류가 발생했는지 자세히 분석해 예측 프로그램을 보완해야 할 대목입니다.
머신러닝 알고리즘을 이용하면 조금 더 정확한 예측을 할 수 있지 않을까 기대합니다.
성주운 기자(seon97jun@gmail.com)
'''
import re
result = re.sub('\([a-zA-Z0-9\._+]+@[a-zA-Z]+\.(com|org|edu|net|co,kr)\)', '',
string) #이메일 패턴을 찾아서 '' 으로 치환
result = re.sub('\n', '', result) #\n을 찾아서 ''으로 치환
# print(result)


# 특수 문자와 숫자 제거
string = '서울의 집값이 올해들어 3.2% 증가했습니다. !!!!'
# 숫자 제거
# 숫자에 해당하는 정규식 객체를 생성
p = re.compile('[0-9]+')
result = p.sub("", string)
# print(result)
# 단어 이외의 문자를 제거
p = re.compile("\W+")
result = p.sub(" ", result)
# print(result)


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

string = "The science of today is the technology of tomorrow. Tomorrow is today."

import nltk
nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

string = "The science of today is the technology of tomorrow. Tomorrow is today."
# 마침표를 기준으로 문장 분할
print(sent_tokenize(string))
# 공백을 기준으로 문장 분할
print(word_tokenize(string))

# 한글 불용어 제거
words_korean = ['구정', '연휴', '명절', '고속도로', '정체']
stopwords = ['구정', '명절']
r = [i for i in words_korean if i not in stopwords]
print(r)

# 영어 불용어 제거 - NLTK
from nltk.corpus import stopwords
words_english = ['chief', 'justice', ' roberts', 'the', 'president', 'and', 'move']
r = [w for w in words_english if not w in stopwords.words('english')]
print(r)

#영문 불용어제거 - sklearn 이용
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
r = [w for w in words_english if not w in ENGLISH_STOP_WORDS]
print(r)

string = "All pyhthonners have pythoned poorly at least once"
# 단어 단위로 토큰화
words = word_tokenize(string)
print(words)

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
string = "All pyhthonners have pythoned poorly at least once"
# 어간 추출기 생성
ps_stemmer = PorterStemmer()
# 단어 단위로 토큰화
words = word_tokenize(string)
print(words)
# 어간 추출
for w in words:
    print(ps_stemmer.stem(w), end = '')
print()

from nltk.stem.lancaster import LancasterStemmer
# 어간 추출기 생성
ls_stemmer = LancasterStemmer()
# 어간 추출
for w in words:
    print(ls_stemmer.stem(w), end='')
print()


import nltk 
nltk.download('averaged_perceptron_tagger')

# 문장 수치화

from nltk import pos_tag
# create text
tweets = ["I am eating a burrito for breakfast", "San Fransico is an awesome city"]
# 단어를 저장할 list
words_tweets = []
# 품사를 저장할 list
tagged_tweets = []
for tweet in tweets:
    tweet_tag = nltk.pos_tag(word_tokenize(tweet))
    # 리스트를 순회하면서 단어와 태그를 list에 삽입
    tagged_tweets.append([tag for (word, tag) in tweet_tag])
    words_tweets.append([word for (word, tag) in tweet_tag])

# 문장을 수치화 - 원핫인코딩: 문장을 하나로 보고 수행
one_hot_multp = preprocessing.MultiLabelBinarizer()
print(one_hot_multi.fit_transform(words_tweets))
print(one_hot_multi.classes_)