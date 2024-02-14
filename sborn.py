import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset("tips")
# print(tips.head())

# 산점도 & 회귀식
# plt.figure(figsize=(8,6)) #캔버스 크기
# sns.regplot(x='total_bill', y='tip', data=tips)
# plt.show()
# hue -> 카테고리별 회귀선
# sns.lmplot(x='total_bill', y='tip', hue='smoker', data=tips)
# plt.show()

# heatmap
flights = sns.load_dataset('flights')
# print(flights)

# index - month // columns - year // passengers 합계 구하는 피벗을 생성
# values는 여러 개 설정 가능
# data = flights.pivot(index='month', columns='year', values='passengers')
# print(data)

# sns.heatmap(data, annot=True, fmt='d')
# plt.show()

# box-plot
# sns.boxplot(x = "day", y = "total_bill", data=tips, hue='smoker')
# plt.show()

# violin plot
# sns.violinplot(x = "day", y = "total_bill", data=tips, hue='smoker')
# plt.show()

# stripplot
# sns.stripplot(x = "day", y = "total_bill", data=tips, hue='smoker')
# plt.show()

# swarmplot()
#sns.set_palette("Set3")
# sns.reset_defaults()
# sns.swarmplot(x = "day", y = "total_bill", data=tips)
# sns.boxplot(x = "day", y = "total_bill", data=tips)
# plt.show()

# pairplot()
# sns.pairplot(tips[["total_bill", "tip", "size"]])
# plt.show()

# plotnine
import plotnine 
import pandas as pd
df = pd.DataFrame({
      'letter':['Alpha', 'Beta', 'Delta', 'Gamma'] * 2,
      'pos':[1, 2, 3, 4] * 2,
      'num_of_letters': [5, 4, 5, 5] * 2
})

# (plotnine.ggplot(df) + plotnine.geom_col(plotnine.aes(x = 'letter', y = 'pos', fill = 'letter')) + plotnine.geom_line(plotnine.aes(x='letter', y ='num_of_letters', color = 'letter'), size = 1) + plotnine.scale_color_hue(l=0.45) + plotnine.ggtitle("Greek Letter Analysis"))
(plotnine.ggplot(df)
+ plotnine.geom_col(plotnine.aes(x='letter', y='pos', fill='letter'))
+ plotnine.geom_line(plotnine.aes(x='letter', y='num_of_letters', color='letter'),size=1)
+ plotnine.scale_color_hue(l=0.45)
+ plotnine.ggtitle('Greek Letter Analysis')
)



import folium 
m = folium.Map(location=[37.572655, 126.973300], zoom_start=15)

folium.Marker(location=[37.572655, 126.973304], popup="KB 국민카드", icon=folium.Icon(icon='cloud')).add_to(m)
folium.Marker(location=[37.572627, 126.973304], popup="KB 국민카드", icon=folium.Icon(color='red')).add_to(m)

m.save('map.html')
