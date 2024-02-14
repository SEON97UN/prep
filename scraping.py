# 검색어 입력 받어서 URL 만들기
from urllib.parse import quote

string = input("검색어를 입력하세요:")
#print(string)
keyword = quote(string)
#print(keyword)
target_URL = "https://www.donga.com/news/search?query={}".format(keyword)
#print(target_URL)

# 첫 화면의 HTML 가져오기
# 정적인 웹의 데이터 가져올 때 사용하는 package => requests
import requests
html = requests.get(target_URL).text
#print(html)


# 검색된 기사 개수 가져오기

# HTML, XML 파싱에 사용하는 패키지
from bs4 import BeautifulSoup

# HTML 텍스트는 메모리에 트리 형태로 펼치기
bs = BeautifulSoup(html, 'html.parser')

# 선택자는 동일한 데이터가 있을 수 있으므로 list
cnt = bs.select('div.cntPageBox > div > span')

'''
for x in cnt:
    print(x.getText())
'''

# 태그 안의 text 가져오기
cnt = cnt[1].getText()

cnt = int(cnt[0 : -1].replace(",", ""))
#print(cnt)

# 페이지 개수 만들기
pageno = int(cnt / 15 + 0.99)
#print(pageno)

# 기사의 링크를 저장할 list
links = []

# 페이지 개수만큼 순회하면서 html을 읽어서 파싱해서 저장
for i in range(0, 4):
    # 기사 링크와 제목이 나오는 페이지를 읽기
    url = "https://www.donga.com/news/search?p={}&query={}&check_news=92&more=1&sorting=3&search_date=1&v1=&v2=".format(str(i*15+1),keyword)
    html = requests.get(url).text
    soup = BeautifulSoup(html, "html.parser")
    linktag = bs.select('#content > div.sch_cont > div.schcont_wrap > div > div:nth-child(2) > div.rightList > span.tit > a')
    for tag in linktag:
        print(tag['href'])
        # a 태그의 href 속성의 값을 가져오기
        links.append(tag['href'])

#print(links)


# 기사 링크를 따라가서 실제 기사를 읽어서 파일에 저장
output_file = open(string + ".txt", 'w', encoding='utf8')

for link in links[1:]:
    #print(link)
    html = requests.get(link).text
    #print(html)
    bs = BeautifulSoup(html, 'html.parser')
    articles = bs.select('div.content')
    for article in articles:
        article.getText()
        # 파일에 기록
        output_file.write(article.getText())
output_file.close()



