import pandas as pd

df=pd.read_csv("wr88125.txt", names=['index', 'year', 'month', 'day', 'min_t', 'average_t', 'max_t', 'rainfall'], sep=";", skipinitialspace=True)
print("Задание 1 выполнено.")
df=df.drop(columns=["index"])
print("Задание 2 выполнено")
df.info(verbose=True)
print('Да, существуют.')
print(str(df.isnull().sum()))
print('В строке max_t больше всего пропущенных значений. Задание 3 выполнено.')
min_count=df.groupby(pd.Grouper(key="year")).count().sum(axis=1).min()
df2=df.groupby(pd.Grouper(key="year")).count().sum(axis=1)
k=df2.keys()
v=df2.values
s = pd.Series(data = k, index = v)
print("Больше всего пропусков за "+str(s[min_count])+" год. Задание 4 выполнено.")
df4=df.groupby(pd.Grouper(key="year"))
range_t=[]
precipitation=[]
chek=0
average_min_t=[]
average_max_t=[]
for i in range(df.shape[0]):
    df["year"][i]=str(df["year"][i])+'-'+str(df["month"][i])+'-'+str(df["day"][i])
    range_t.append(round((df["max_t"][i]-df["min_t"][i]), 2))
    if df["rainfall"][i]!=0:
        chek=0
    else:
        chek+=1
    if df["average_t"][i]>27:
        average_max_t.append(df["year"][i])
    if df["average_t"][i]<-30 and chek>3:
        average_min_t.append(df["year"][i])
    precipitation.append(chek)
print("Задание 6 выполнено")
df=df.drop(columns=["month", "day"])
df['year'] = pd.to_datetime(df['year'])
df=df.rename(columns={'year': 'date', 'min_t': 'min_t', 'average_t':'average_t', 'max_t': 'max_t', 'rainfall' : 'rainfall'})
print("Задание 5 выполнено")
df3 = pd.DataFrame({"range_t": range_t, "precipitation": precipitation})
df=pd.concat([df, df3], axis=1)
print("Самый длинный период засухи: "+str(df["precipitation"].max())+" дней")
max_t=df4["average_t"].mean().max()
min_t=df4["average_t"].mean().min()
df_average_t=df4["average_t"].mean()
k=df_average_t.keys()
v=df_average_t.values
year_average_t = pd.Series(data = k, index = v)
print("Самый тёплый год был: "+str(year_average_t[max_t])+". Самый холодный год был: "+ str(year_average_t[min_t]))
rain=df4["rainfall"].sum().max()
df_rain=df4["rainfall"].sum()
k=df_rain.keys()
v=df_rain.values
year_rain = pd.Series(data = k, index = v)
print("Даты, когда средняя температура воздуха ниже -30 оС"+ str(average_min_t))
print("Даты, когда средняя температура воздуха выше 27оС и количество дней без осадков больше 3"+ str(average_max_t))