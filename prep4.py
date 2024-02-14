import pandas as pd
import numpy as np


# Use data provided by scikit-learn
from sklearn import datasets

# iris = datasets.load_iris()

# print(type(iris))   # data -> feature // target -> label
# print(iris.keys())
# print(iris.data)

# # read text file 
# df_fwf = pd.read_fwf('c:\data\data_fwf.txt', widths=(10, 2, 5), names = ("날짜","이름","가격"), encoding = 'utf-8')
# print(df_fwf)

# # read csv file
# df_item = pd.read_csv('c:\data\item.csv')
# print(df_item)
# print(df_item.info())

# df_tx = pd.read_csv('c:\\Users\\USER\\Documents\\suwon_taxation.csv', header=1, encoding='cp949')
# print(df_tx)

# 데이터의 개수를 알거나, 예외처리 필요
# i = 0
# while True:
#   try:
#     df_good = pd.read_csv('c:\data\good.csv', header=None, nrows = 2, skiprows = i*2)
#     print(df_good)
#     i = i + 1
#   except: 
#     break

# df_good = pd.read_csv('c:\data\good.csv', header=None, chunksize=2)
# for piece in df_good:
#       print(piece)

# df_gapminder = pd.read_csv('c:\\data\gapminder.tsv', sep='\t')
# print(df_gapminder.head())


# df_tx = pd.read_csv('c:\\Users\\USER\\Documents\\suwon_taxation.csv', header=0, encoding='cp949')
# print(df_tx.head())

# i = 0
# while True:
#   try:
#     df_tx = pd.read_csv('c:\\Users\\USER\\Documents\\suwon_taxation.csv', header=0, nrows=5, skiprows=i*2, encoding='cp949')
#     print(df_tx)
#     i = i+1
#   except:
#     break

# df_tx.to_csv('test.csv', index=False)
# data = pd.read_csv('test.csv')
# print(data.head())

# excel.xlsx 파일을 읽기
# df = pd.read_excel('c:\\data\\excel.xlsx')
# print(df.info())
# print(df)

# # excel.xlsx 파일 저장 -> 현재 디렉토리에 저장됨
# writer = pd.ExcelWriter('sample.xlsx', engine = 'xlsxwriter')
# df.to_excel(writer, sheet_name='excel')
# # print(dir(writer))
# writer.close()

# # read_html은 list를 리턴하므로 인덱스를 이용해서 원하는 테이블 선택해야함
# li = pd.read_html('https://ko.wikipedia.org/wiki/%EC%9D%B8%EA%B5%AC%EC%88%9C_%EB%82%98%EB%9D%BC_%EB%AA%A9%EB%A1%9D')
# print(li[0])

import urllib.request
res = urllib.request.urlopen('https://www.kakao.com')
# print(res.read())

# # 한글을 인코딩
# from urllib.parse import quote
# keyword = quote("삼성전자") 

# srch = urllib.request.urlopen('https://www.joongang.co.kr/search?keyword=' +keyword)
# print(srch.read())

# import requests
# resp = requests.get("http://httpbin.org/get")
# print(resp.text)

# para = {"id":"seon97un", "name":"seongjun", "age":28}
# resp = requests.post("http://httpbin.org/get", data = para)
# print(resp.text)

# imageurl = "https://png.pngtree.com/thumb_back/fh260/background/20230609/pngtree-three-puppies-with-their-mouths-open-are-posing-for-a-photo-image_2902292.jpg"
# filename = "img1.jpg"

# try:
#   resp = requests.get(imageurl)
#   with open(filename, "wb") as h:
#     img = resp.content
#     h.write(img)
# except Exception as e:
#   print(e)


# df = pd.read_json("http://swiftapi.rubypaper.co.kr:2029/hoppin/movies?version=1&page=1&count=30&genreId=&order=releasedateasc")
# print(type(df))
# hoppin = df["hoppin"]  
# movies = hoppin["movies"]
# movie = movies["movie"]     
# for item in movie:
#     print(item["title"] + ":" + item["ratingAverage"])             

    
import urllib.request
# xml 파싱
import xml.etree.ElementTree as et
# url = " https://www.chosun.com/arc/outboundfeeds/rss/?outputType=xml"
# request = urllib.request.Request(url)
# response = urllib.request.urlopen(request)
# print(response.read())


# # 메모리에 펼치기
# tree = et.parse(response)

# # 루트 찾기
# xroot = tree.getroot()
# print(xroot)


import requests
import bs4
resp = requests.get("https://www.nytimes.com/")
html = resp.text
# print(html)

# html 파싱
bs = bs4.BeautifulSoup(html, 'html.parser')
tags = bs.select("div.smartphone.tablet > div > div:nth-child(1) > div:nth-child(1) > div > div:nth-child(1) > div > div > div > div > div.css-1432v6n.e17qa79g0 > div > section:nth-child(1) > a > div > div > p")
for tag in tags:
    print(tag.getText())