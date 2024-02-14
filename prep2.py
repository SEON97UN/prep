import numpy as np
import pandas as pd

ar = np.array([10, 2, 3, 4, np.nan, 32, 42])
# 배열에 None 데이터가 있는지 확인
# print(np.isnan(ar)) # 결측치 확인

# # 결측치 제외하고 가져오기
# result = ar[np.logical_not(np.isnan(ar))]
# print(result)

# # 결측치를 제외하고 가져오기 - 결측치 제거하지 않고 수행하면 nan이 포함됨
# result = result[np.logical_not(np.isnan(result > 10))]
# print(result)

# 3의 배수이거나 4의 배수인 데이터만 추출
# ar = np.array([10, 2, 3, 4, np.nan, 32, 42])
# result = ar[np.logical_not(np.isnan(ar))]
# rslt = result[(result % 4 == 0) | (result % 3 == 0)]
# print(rslt)

ar = np.array([2, 1, 3, 4, 5, 1, 2, 3])
print(np.unique(ar))

arr = np.array([2, 1, 3, 4, np.nan])
arr.sort()
print(arr)

# numpy의 sort는 리넡을 하고 ndarray의 sort는 리턴하지 않음
# 내림차순 정렬은 numpy의 sort를 이용하면 됨
print(np.sort(arr)[::-1])

# 2차원 배열은  numpy에서 sort를 잘 사용하지 않음
# 데이터의 인덱스가 깨져버리기 때문
matrixx = np.array([[1, 3, 4, 2],[9, 4, 1, 8]])
matrixx.sort(axis=0)
print(matrixx)

np.save('ar.npy', ar)
x = np.load('ar.npy')
print(x)