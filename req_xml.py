import urllib.request
# xml 파싱
import xml.etree.ElementTree as et
url = " https://www.chosun.com/arc/outboundfeeds/rss/?outputType=xml"
request = urllib.request.Request(url)
response = urllib.request.urlopen(request)

import xml.etree.ElementTree as et

# 응답이 비어 있거나 None인지 확인
if response:
    try:
        tree = et.parse(response)
        # 파싱된 XML 트리를 처리 계속
    except et.ParseError as e:
        print("XML 파싱 오류:", e)
else:
    print("빈 XML 데이터 또는 응답이 None입니다")
xroot = tree.getroot()
print(xroot)