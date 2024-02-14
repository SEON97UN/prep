import numpy as np
import pandas as pd

# item = pd.read_csv('c:\data_all/item.csv')
# # print(item)


# # numpy나 pandas의 대다수의 메서드는 원본을 변경하지 않고 수정해서 리턴
# # pandas의 DataFrame에서는 inplace 옵션이 있는 경우 이 옵션에 True를 설정하면 원본을 수정
# names = columns={"code":"코드", "manufacture":"원산지", "name":"이름", "price":"가격"}
# item.rename(columns=names, inplace=True)
# # print(item)

# # item에서 코드를 가져와서 인덱스로 설정 - 코드는 컬럼으로 존재
# item.index = item.코드
# # print(item)

# # set_index를 이용하면 컬럼에서 제거되고 인덱스로 설정
# item.set_index("코드", inplace=True)
# # print(item.set_index("코드"))

# # 인덱스를 다시 컬럼으로 만들고 0부터 시작하는 일련번호 인덱스 생성
# item = item.reset_index()
# # print(item)

# # 행이나 열 삭제
# # 2행 삭제
# print(item.drop([1], axis=0))
# # code 열 삭제
# print(item.drop(["코드"], axis=1))

# 데이터 추가 및 삭제
item = pd.read_csv('c:\data_all/item.csv')
item.info()

# 컬럼 추가
item['description'] = '과일'
# print(item)

# 컬럼 수정 - list는 순서대로 삽입
item['description'] = ['사과', '수박', '참외', '바나나', '레몬', '망고']
# print(item)

# Series나 dict는 인덱스나 키 이름으로 대입
item['description'] = {0:'사과', 1:'수박', 2:'참외', 3:'바나나', 5:'레몬', 4:'망고'}
# print(item)

# 행 추가
item.loc[6] = [7, 'korea', 'grape', 3000, '포도']
# print(item)

# 특정 셀 수정 - 앞에 인덱스 설정, 뒤에 컬럼이름
item.loc[6, 'name'] = 'fig'
# print(item)

# DataFrame 연산
item1 = {"1" : {'price':1000},
         "2" : {'price':2000},
         }
item2 = {"1" : {'price':1000},
         "3" : {'price':3000},
         }
df1 = pd.DataFrame(item1).T
# print(df1)
df2 = pd.DataFrame(item2).T
# print(df2)

# print(df1 + 200) # 200을 df1의 개수만큼 복제 해서 연산

# 존재하지 않는 인덱스의 결과는 NaN
print(df1 + df2)

# 한쪽에만 존재하는 인덱스에 기본값을 설정해서 연산 수행
print(df1.add(df2, fill_value=0))

# Series와 연산할 때 axis = 0을 설정하면 행 단위로 연산 수행
print(df1.add(df2, axis=0))