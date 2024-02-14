import requests
import bs4
resp = requests.get("https://web.kma.go.kr/eng/weather/forecast/current_korea.jsp")
html = resp.text
# print(html)

# html 파싱
bs = bs4.BeautifulSoup(html, 'html.parser')
stn = []
wthr = []

tags = bs.select("td:nth-child(1)")
for tag in tags:
    stn.append(tag.getText())
tags = bs.select("td:nth-child(2)")
for tag in tags:
    wthr.append(tag.getText())


for i in range(len(stn)):
    print(stn[i] + ":" + wthr[i])


