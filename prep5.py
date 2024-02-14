import pandas as pd
import requests
import json

# url 만들기
url =  'https://dapi.kakao.com/v2/local/search/category.json?category_group_code=PM9&rect=126.95,37.55,127.0,37.60'
# 헤더 설정
headers = {'Authorization':'KakaoAK {}'.format('a7aab0b2b25be6c4f710da6af1e91c9b')}
data = requests.post(url, headers=headers)
print(data)
print(data.text)

# JSON 문자열을 Python의 자료구조로 변경
result = json.loads(data.text)
print(result)
print(type(data.text))
print(type(result))

# documents 키의 데이터 가져오기
documents = result['documents']
# print(documents)

for doc in documents:
    print(doc['place_name'], doc['address_name'])


import pymysql
#from pymysql import connect
con = pymysql.connect(host = '127.0.0.1',
                       port = 3306, 
                       user ='seon97un', 
                       password = '9703', 
                       db = 'seon97un', 
                       charset = 'utf8')
print(con)
# SQL을 실행하기 위한 객체 생성
#con.close()


# 테이블 생성 구문 실행
cursor = con.cursor()
cursor.execute("DROP TABLE IF EXISTS phamacy")
cursor.execute("create table phamacy(placename varchar(30), addressname varchar(200))")

#파싱한 데이터 순회
for doc in documents:
  cursor.execute("insert into phamacy(placename, addressname) values(%s, %s)", 
                 (doc["place_name"], doc["address_name"]))
con.commit()
con.close

