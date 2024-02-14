# import seaborn as sns
# anscombe = sns.load_dataset("anscombe")
# # print(anscombe.head())
# print(anscombe['dataset'].unique())

# # dataset 별 분리
# dataset_1 = anscombe[anscombe['dataset'] == 'I']
# dataset_2 = anscombe[anscombe['dataset'] == 'II']
# dataset_3 = anscombe[anscombe['dataset'] == 'III']
# dataset_4 = anscombe[anscombe['dataset'] == 'IV']

# '''
# print(dataset_1['x'].mean())
# print(dataset_2['x'].mean())
# print(dataset_3['x'].mean())
# print(dataset_4['x'].mean())

# print(dataset_1['x'].std())
# print(dataset_2['x'].std())
# print(dataset_3['x'].std())
# print(dataset_4['x'].std())
# '''

# # visualization
# # import matplotlib.pyplot as plt
# fig = plt.figure()

# axes1 = fig.add_subplot(2, 2, 1)
# axes2 = fig.add_subplot(2, 2, 2)
# axes3 = fig.add_subplot(2, 2, 3)
# axes4 = fig.add_subplot(2, 2, 4)

# axes1.plot(dataset_1['x'], dataset_1['y'], 'o')
# axes2.plot(dataset_2['x'], dataset_2['y'], 'o')
# axes3.plot(dataset_3['x'], dataset_3['y'], 'o')
# axes4.plot(dataset_4['x'], dataset_4['y'], 'o')

# axes1.set_title("dataset_1")
# axes2.set_title("dataset_2")
# axes3.set_title("dataset_3")
# axes4.set_title("dataset_4")

# # plt.show()

