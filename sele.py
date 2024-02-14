from selenium import webdriver
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import bs4
import time

# os.environ['webdriver.chrome.driver'] = 'C:\\Users\\USER\\Downloads\\chromedriver-win64\\chromedriver-win64'
# driver = webdriver.Chrome()

# # 사이트 접속
# driver.get("http://www.kakao.com")

# # html 읽어오기 
# html = driver.page_source
# print(html)
# while(True):
#     pass


# 카카오 로그인

os.environ['webdriver.chrome.driver'] = 'C:\\Users\\USER\\Downloads\\chromedriver-win64\\chromedriver-win64'
driver = webdriver.Chrome()

# 사이트 접속
driver.get("https://accounts.kakao.com/login/?continue=https%3A%2F%2Flogins.daum.net%2Faccounts%2Fksso.do%3Frescue%3Dtrue%26url%3Dhttps%253A%252F%252Fwww.daum.net#login")

# 5초간 대기
driver.implicitly_wait(5)

userid = "sean703@naver.com"
password = "vpdlzl703"

'''
driver.find_element(By.XPATH, '//*[@id="loginId--1"]').send_keys(userid)
driver.find_element(By.XPATH, '//*[@id="password--2"]').send_keys(password)
driver.find_element(By.XPATH, '//*[@id="mainContent"]/div/div/form/div[4]/button[1]"]').click()
'''

#driver.execute_script("document.getElementsByName('loginId')[0].value=\'" + userid+ "\'")
driver.find_element(By.XPATH, '//*[@id="loginId--1"]').send_keys(userid)
driver.find_element(By.XPATH, '//*[@id="password--2"]').send_keys(password)
#driver.execute_script("document.getElementsByName('password')[0].value=\'" + password+ "\'")
driver.find_element(By.XPATH, '//*[@id="mainContent"]/div/div/form/div[4]/button[1]').click()

# html 읽어오기 
html = driver.page_source
# print(html)


while(True):
    pass


# # 유튜브 스크롤
# os.environ['webdriver.chrome.driver'] = 'C:\\Users\\USER\\Downloads\\chromedriver-win64\\chromedriver-win64'
# driver = webdriver.Chrome()

# # 사이트 접속
# driver.get("https://www.youtube.com/results?search_query=%EC%82%BC%ED%94%84%EB%A1%9C")

# # 5초간 대기
# time.sleep(5)

# i = 0

# body = driver.find_element(By.TAG_NAME, 'body')
# while i < 10:
#     body.send_keys(Keys.PAGE_DOWN)
#     time.sleep(2)
#     i = i + 1


# # html 읽어오기 
# html = driver.page_source
# # print(html)
# while(True):
#     pass

