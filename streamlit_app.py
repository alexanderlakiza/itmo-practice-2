import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

st.title("Предсказание спроса на товары с помощью машинного обучения")
st.markdown("## В рамках текущей задачи мы будем предсказывать спрос на товар в магазине")
st.markdown("### Датасет взят с [соревнования](https://www.kaggle.com/competitions/demand-forecasting-kernels-only) "
            "на Kaggle")
st.markdown("В учебных целях я решил упростить исходную задачу. Я буду строить прогноз спроса только для одной "
            "пары `магазин` + `товар`, а не для 50, и прогноз я буду строить не на первые 3 месяца 2018 года "
            "(как в соревновании), а для всего 2017 года (чтобы была возможность сразу взглянуть на метрики)")

st.markdown("---")
st.markdown("## Краткий анализ данных")
train_df = pd.read_parquet('data/model_datasets/train.parquet')
test_df = pd.read_parquet('data/model_datasets/test.parquet')

X_train, y_train = train_df.drop('target', axis=1), train_df['target']
X_test, y_test = test_df.drop('target', axis=1), test_df['target']

st.markdown("Исходный датасет")
df = pd.read_csv('data/train.csv')
st.dataframe(df.sample(10))
bar_fig = px.bar(pd.pivot_table(df.rename(columns={'store': 'store id',
                                                   'item': 'item id'}), values='sales',
                                columns='store id',
                                index='item id',
                                aggfunc='sum'), title="Распределение суммы продаж по товарам и магазинам",
                 template='plotly_dark')
st.plotly_chart(bar_fig)
st.markdown("Мы немного похимичили над датасетом. Посмотрим, что из себя сейчас представляет датасет для обучения")
st.dataframe(train_df.head(10).astype(int).reset_index().drop('index', axis=1))

st.markdown("Распределение продаж по дням, которые нам необходимо предсказывать")


def plot_train_test_lines():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pd.date_range("2014-01-01", "2016-12-31", freq='D'),
                             y=y_train,
                             name='Обучающая<br>выборка',
                             mode='lines',
                             line={'color': '#636EFA'}))

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=y_test,
                             name='Тестовая<br>выборка',
                             mode='lines',
                             line={'color': '#EF553B'}))

    fig.update_layout(title='Продажи<br>'
                            '<b>Обучающая выборка</b> vs <b>Тестовая выборка</b>')
    fig.update_xaxes(title='Дата', tickformat="%d %b %Y")
    fig.update_yaxes(title='Продажи за день')
    return fig


st.plotly_chart(plot_train_test_lines())
st.markdown('---')
st.markdown("## Результаты предсказания спроса")
st.markdown("### Результаты метрик")
metrics_df = pd.read_parquet('data/model_results/metrics.parquet')
st.dataframe(metrics_df)
st.markdown("Как мы видим в таблице, лучше всего себя проявила линейная регрессия")
st.caption("Стоит учесть тот факт, что в таблице сверху указаны значения метрик для ежедневного предсказания "
           "на год вперёд")

pred_df = pd.read_parquet('data/model_results/predicts.parquet')


def plot_train_test_pred_lines():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pd.date_range("2014-01-01", "2016-12-31", freq='D'),
                             y=y_train,
                             name='Обучающая<br>выборка',
                             mode='lines',
                             line={'color': '#636EFA'}))

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=y_test,
                             name='Тестовая<br>выборка',
                             mode='lines',
                             line={'color': '#EF553B'}))

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=pred_df['lin_reg'],
                             name='Линейная<br>регрессия',
                             mode='lines',
                             line={'color': '#00CC96'}))

    fig.update_layout(title='Продажи<br>'
                            '<b>Обучающая выборка</b> vs <b>Тестовая выборка</b> vs <b>Лучший прогноз</b>')
    fig.update_xaxes(title='Дата', tickformat="%d %b %Y")
    fig.update_yaxes(title='Продажи за день')
    return fig


st.plotly_chart(plot_train_test_pred_lines())

st.markdown("### Взглянем на сравнение всех моделей прогноза с реальными данными")


def plot_test_pred_lines():
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=y_test,
                             name='Тестовая<br>выборка',
                             mode='lines'))

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=pred_df['lin_reg'],
                             name='Линейная<br>регрессия',
                             mode='lines'))

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=pred_df['lin_reg_poly'],
                             name='Полиномиальная<br>Линейная<br>регрессия',
                             mode='lines'))

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=pred_df['rf'],
                             name='Случайный<br>лес',
                             mode='lines'))
    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=pred_df['cat'],
                             name='Градиентный<br>бустинг',
                             mode='lines'))

    fig.update_layout(title='<b>Прогнозы и реальное потребление</b>')
    fig.update_xaxes(title='Дата', tickformat="%d %b %Y")
    fig.update_yaxes(title='Продажи за день')
    return fig


st.plotly_chart(plot_test_pred_lines())


def plot_test_and_one_pred(model, name, colour):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=y_test,
                             name='Тестовая<br>выборка',
                             mode='lines'))

    fig.add_trace(go.Scatter(x=pd.date_range("2017-01-01", "2017-12-31", freq='D'),
                             y=pred_df[model],
                             name=name,
                             mode='lines',
                             line={'color': colour}))

    fig.update_layout(title=f'<b>Реальное потребление vs {name}</b>')
    fig.update_xaxes(title='Дата', tickformat="%d %b %Y")
    fig.update_yaxes(title='Продажи за день')
    return fig


colours = px.colors.qualitative.Plotly
st.plotly_chart(plot_test_and_one_pred('lin_reg', 'Линейная регрессия', colours[1]))
st.plotly_chart(plot_test_and_one_pred('lin_reg_poly', 'Полиномильаная линейная<br>регрессия', colours[2]))
st.plotly_chart(plot_test_and_one_pred('rf', 'Случайный лес', colours[4]))
st.plotly_chart(plot_test_and_one_pred('cat', 'Градиентный бустинг', colours[3]))
st.plotly_chart(plot_test_and_one_pred('lstm', 'LSTM', colours[6]))


def smape(a, f):
    return 1 / len(a) * np.sum(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)) * 100)


def root_mean_squared_error(true, pred):
    return mean_squared_error(true, pred) ** (1 / 2)


target_df = pd.DataFrame(y_test)
target_df['date'] = pd.date_range('2017-01-01', '2017-12-31', freq='D')
month_pred = pred_df.groupby([pd.Grouper(freq='MS', key='date')]).sum().reset_index()
month_target = target_df.groupby([pd.Grouper(freq='MS', key='date')]).sum().reset_index()

month_metrics = pd.concat(
    [month_pred.drop('date', axis=1).apply(lambda x: root_mean_squared_error(month_target['target'], x), axis=0),
     month_pred.drop('date', axis=1).apply(lambda x: mean_absolute_error(month_target['target'], x), axis=0),
     month_pred.drop('date', axis=1).apply(lambda x: mean_absolute_percentage_error(month_target['target'], x), axis=0),
     month_pred.drop('date', axis=1).apply(lambda x: smape(month_target['target'], x), axis=0)], axis=1)

month_metrics.columns = ['RMSE', 'MAE', 'MAPE', 'sMAPE']

st.markdown("### Взглянем на метрики, если детализация прогноза будет месячная, а не дневная")
st.dataframe(month_metrics)

st.markdown("### Теперь посмотрим на метрики при годовой детализации прогноза")
year_pred = pred_df.groupby([pd.Grouper(freq='Y', key='date')]).sum().reset_index()
year_target = target_df.groupby([pd.Grouper(freq='Y', key='date')]).sum().reset_index()

year_metrics = pd.concat(
    [year_pred.drop('date', axis=1).apply(lambda x: mean_absolute_error(year_target['target'], x), axis=0),
     year_pred.drop('date', axis=1).apply(lambda x: mean_absolute_percentage_error(year_target['target'], x), axis=0),
     year_pred.drop('date', axis=1).apply(lambda x: smape(year_target['target'], x), axis=0)], axis=1)

year_metrics.columns = ['MAE', 'MAPE', 'sMAPE']
st.dataframe(year_metrics)

st.markdown("### Как мы видим, даже при разной детализации прогноза лучше всего "
            "себя проявляет линейная регрессия!!!")
st.balloons()
