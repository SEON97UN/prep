import numpy as np

# Slicing
ar = np.arange(10)
matrix = ar.reshape((2,-1))

# print(matrix[1][1:3])
# print(matrix[1]) # 1행 전체
# print(matrix[:, 1]) # 2열 전체

# 데이터를 복제 -> br이 가리키도록 함 
# 원본 데이터도 영향을 받지 않음
br = ar[0:4].copy()
# print(br)
br[0] = 42
# print(br[0])
# print(ar[0])



# Fancy Indexing
# 데이터를 복제함
br = ar[[1, 3, 5, 7]]
# print(br)
br[0] = 15700
# print(br[0])
# print(ar[0])

matrix = ar.reshape([2, -1])
# print(matrix)
# 이차원 배열에서 list를 이용해서 행 번호나 열 번호를 지정하면 이차원 배열이 만들어짐
# print(matrix[:, [0]])
# numpy의 ndarray나 pandas의 DataFrame에서 
# 하나의 열을 선택할 때, list로 설정하는 경우는 구조를 유지하기 위함


# 브로드캐스트 연산
print(matrix + 10)
data = np.array([100, 200, 300, 400, 500])
print(matrix + data)
# numpy의 ndarray와 논리연산을 수행하면 bool 배열이 만들어짐
print(ar == 3)
# 인덱스에 bool 배열을 대입하면 True인 데이터만 추출됨
print(ar[ar % 2 == 1])    # 홀수냐? True인 값만 나와라!
print(ar[(ar % 2 == 1) | (ar % 4 == 0)]) # 홀수 OR 4의 배수
