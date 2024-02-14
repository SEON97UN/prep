import urllib.request
url = " https://www.chosun.com/arc/outboundfeeds/rss/?outputType=xml"
request = urllib.request.Request(url)
response = urllib.urlopen(request)
print(response.read())