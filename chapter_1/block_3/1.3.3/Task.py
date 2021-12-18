#Импартируем библиотеку pandas
import pandas as pd

#Импортируем таблицу из внешнего файла, добовляем названия к столбцам, все пробелы помечаем как отсуствующие данные
df=pd.read_csv("wr88125.txt", names=['index', 'year', 'month', 'day', 'min_t', 'average_t', 'max_t', 'rainfall'], sep=";", skipinitialspace=True)
print("Задание 1 выполнено.")
# Удаляем столбец index
df=df.drop(columns=["index"])
print("Задание 2 выполнено")
#Используя метод info отвечаем на вопросы задания 3
df.info(verbose=True)
print('Да, существуют.')
print(str(df.isnull().sum()))
print('В строке max_t больше всего пропущенных значений. Задание 3 выполнено.')
#По заданию 4 определяем год с наибольшим количеством пропусков
#Определяем наименьшее количество значений по всем столбцам за каждый год
min_count=df.groupby(pd.Grouper(key="year")).count().sum(axis=1).min()
df2=df.groupby(pd.Grouper(key="year")).count().sum(axis=1)
#Перегрупперуем серию что бы по имеющемуся значению пропусков вывести значение года.
k=df2.keys()
v=df2.values
s = pd.Series(data = k, index = v)
print("Больше всего пропусков за "+str(s[min_count])+" год. Задание 4 выполнено.")
#Создаём группировку по годам для последующих заданий
df4=df.groupby(pd.Grouper(key="year"))
range_t=[]
precipitation=[]
chek=0
average_min_t=[]
average_max_t=[]
#Создаём цикл для перебора всех строк имеющихся данных
for i in range(df.shape[0]):
    #В каждой строке в столбец year вставляем строковое значение даты в формате гггг-мм-дд для задания 5
    df["year"][i]=str(df["year"][i])+'-'+str(df["month"][i])+'-'+str(df["day"][i])
    #Вычисляем размах суточных температур по заданию 6
    range_t.append(round((df["max_t"][i]-df["min_t"][i]), 2))
    #Вычисляем периоды без осадков по заданию 6
    if df["rainfall"][i]!=0:
        chek=0
    else:
        chek+=1
    #Определяем даты со средняя температура воздуха выше 27оС и количество дней без осадков больше 3 по заданию 9
    if df["average_t"][i]>27:
        average_max_t.append(df["year"][i])
    # Определяем даты со средняя температура воздуха ниже -30 оС по заданию 9
    if df["average_t"][i]<-30 and chek>3:
        average_min_t.append(df["year"][i])
    precipitation.append(chek)
print("Задание 6 выполнено")
# Удаляем колонки month и day по заданию 5
df=df.drop(columns=["month", "day"])
#Преобразуем первый столбец к типу datatime по заданию 5
df['year'] = pd.to_datetime(df['year'])
#Переименовываем первый столбец по заданию 5
df=df.rename(columns={'year': 'date', 'min_t': 'min_t', 'average_t':'average_t', 'max_t': 'max_t', 'rainfall' : 'rainfall'})
print("Задание 5 выполнено")
#Добавляем информацию об продолжительность дней без осадков и размахе температур в таблицу
df3 = pd.DataFrame({"range_t": range_t, "precipitation": precipitation})
df=pd.concat([df, df3], axis=1)
print("Самый длинный период засухи: "+str(df["precipitation"].max())+" дней")
#Определяем самую низкую и самую высокую среднегодовую температуру по заданию 8
max_t=df4["average_t"].mean().max()
min_t=df4["average_t"].mean().min()
df_average_t=df4["average_t"].mean()
#Перегрупперуем серию что бы по имеющемуся значению вывести значение года.
k=df_average_t.keys()
v=df_average_t.values
year_average_t = pd.Series(data = k, index = v)
print("Самый тёплый год был: "+str(year_average_t[max_t])+". Самый холодный год был: "+ str(year_average_t[min_t]))
#Определяем самый дождливый год по заданию 8
rain_max=df4["rainfall"].sum().max()
rain_min=df4["rainfall"].sum().min()
df_rain=df4["rainfall"].sum()
#Перегрупперуем серию что бы по имеющемуся значению вывести значение года.
k=df_rain.keys()
v=df_rain.values
year_rain = pd.Series(data = k, index = v)
print("Самый дождливый год: "+str(year_rain[rain_max]))
print("Самый Засушливый год: "+str(year_rain[rain_min]))
print("Даты, когда средняя температура воздуха ниже -30 оС"+ str(average_min_t))
print("Даты, когда средняя температура воздуха выше 27оС и количество дней без осадков больше 3"+ str(average_max_t))