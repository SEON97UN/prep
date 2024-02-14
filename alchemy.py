import pandas as pd
import pymysql
from sqlalchemy import create_engine

# 연결
connect = create_engine('mysql+mysqldb://seon97un:9703@localhost/seon97un')
df = pd.read_sql_table('dbms', connect)
print(df)

# 1. 데이터 프레임에서 데이터의 선택
# 1) 열 선택
# 데이터프레임['컬럼이름'] 또는 데이터프레임.컬럼이름
# 데이터프레임.컬럼이름 으로 접근할 때는 컬럼이름이 반드시 문자열이어야 합니다
# 하나의 컬럼이름을 이용해서 접근하면 Series로 리턴
# 2) 행 선택
# loc[인덱스 이름]으로 접근
# iloc[정수형 위치 인덱스]로 접근
# Series로 리턴
# 3) 셀 선택
# [컬럼이름][인덱스이름]의 형태로 접근
# loc[인덱스이름, 컬럼이름]
# iloc[행 위치 인덱스, 열 위치 인덱스]
# 4) 다중 선택
# list를 이용해서 선택
# list를 이용해서 선택하면 DataFrame이 리턴됨