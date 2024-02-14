import pandas as pd
# price = pd.Series([1000, 3000, 2000, 4000])
# print(price)
# print(price[1])

# price.index = ["사탕", "과자", "음료수", "과일"]
# print(price)
# print(price["사탕"])    # indexer -> [""]

# # 데이터의 참조를 가져온 것 -> x 데이터를 변경하면 원본인 price의 데이터도 수정
# x = price["사탕":"음료수"]   # from "사탕" to "음료수" 
# x["사탕"] = 800
# print(x)
# print(price)

# print("===============================================================")

# # 데이터를 복제해 온 것 -> 데이터를 변경해도 원본 price의 데이터 변경 X
# y = price[["사탕","음료수"]]   # fancy indexing -> "사탕" and "음료수" 
# y["사탕"] = 2000
# print(y)
# print(price)

# import numpy as np
# s1 = pd.Series([100, 200, 300, np.nan], index = ["사과", "배", "한라봉", "천혜향"])
# print(s1)
# s2 = pd.Series([100, 200, 300, 500], index = ["사과", "한라봉", "천혜향", "무화과"])
# print(s2)

# print(s1 + s2)    # Series는 index를 기준으로 계산함


# DataFrame 생성
source = {
  "code" : [1,2,3,4],
  "name" : ["카리나", "지젤", "닝닝", "윈터"],
  "age"  : [23,22,34,21]
}
df = pd.DataFrame(source)
# print(df)

# print(df.head(2))   # 앞에서 2개
# print(df.tail(2))   # 뒤에서 2개

# df.info()

print(df.index)
print(df.values)  # numpy의 ndarray는 ,가 나오지 않음
print(type(df.values))