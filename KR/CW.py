from sklearn.datasets import *
import streamlit as st
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, NuSVC, LinearSVC, OneClassSVM, SVR, NuSVR, LinearSVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from sklearn.metrics import roc_curve, roc_auc_score


@st.cache
def load_data():
    '''
    Загрузка обучающей выборки
    '''
    pre_data = pd.read_csv('C:\DataSet\Heart_train.csv', sep=",")

    data1 = pre_data.drop(["age"], axis=1)
    data2 = data1.drop(["sex"], axis=1)
    data3 = data2.drop(["trestbps"], axis=1)
    data4 = data3.drop(["chol"], axis=1)
    data5 = data4.drop(["fbs"], axis=1)
    data6 = data5.drop(["restecg"], axis=1)
    data_X = data6.drop(["target"], axis=1)
    data_Y = pre_data[["target"]]

    sc = MinMaxScaler()
    data_X_sc = sc.fit_transform(data_X)
    data_Y_sc = data_Y

    return data_X_sc, data_Y_sc, data6.shape[0], data6

def load_test():
    '''
    Загрузка тестовой выборки
    '''
    pre_data = pd.read_csv('C:\DataSet\Heart_test.csv', sep=",")

    data1 = pre_data.drop(["age"], axis=1)
    data2 = data1.drop(["sex"], axis=1)
    data3 = data2.drop(["trestbps"], axis=1)
    data4 = data3.drop(["chol"], axis=1)
    data5 = data4.drop(["fbs"], axis=1)
    data6 = data5.drop(["restecg"], axis=1)
    data_X = data6.drop(["target"], axis=1)
    data_Y = pre_data[["target"]]

    sc = MinMaxScaler()
    data_X_sc = sc.fit_transform(data_X)
    data_Y_sc = data_Y

    return data_X_sc, data_Y_sc, data6.shape[0], data6

st.title('ИУ5-65Б Камалов Марат КР')

data_load_state = st.text('Загрузка данных...')
data_X, data_Y, data_len, data  = load_data()
test_X, test_Y, test_len, test  = load_test()
data_load_state.text('Данные загружены!')

clas_X_train = data_X
clas_X_test = test_X
clas_Y_train = data_Y
clas_Y_test = test_Y

# Модели
clas_models = {'LogR': LogisticRegression(),
               'KNN_10':KNeighborsClassifier(n_neighbors=10),
               'SVC':SVC(probability=True),
               'Tree':DecisionTreeClassifier(),
               'RF':RandomForestClassifier(),
               'GB':GradientBoostingClassifier()}


class MetricLogger:

    def __init__(self):
        self.df = pd.DataFrame(
            {'metric': pd.Series([], dtype='str'),
             'alg': pd.Series([], dtype='str'),
             'value': pd.Series([], dtype='float')})

    def add(self, metric, alg, value):
        """
        Добавление значения
        """
        # Удаление значения если оно уже было ранее добавлено
        self.df.drop(self.df[(self.df['metric'] == metric) & (self.df['alg'] == alg)].index, inplace=True)
        # Добавление нового значения
        temp = [{'metric': metric, 'alg': alg, 'value': value}]
        self.df = self.df.append(temp, ignore_index=True)

    def get_data_for_metric(self, metric, ascending=True):
        """
        Формирование данных с фильтром по метрике
        """
        temp_data = self.df[self.df['metric'] == metric]
        temp_data_2 = temp_data.sort_values(by='value', ascending=ascending)
        return temp_data_2['alg'].values, temp_data_2['value'].values

    def plot(self, str_header, metric, ascending=True, figsize=(5, 5)):
        """
        Вывод графика
        """
        array_labels, array_metric = self.get_data_for_metric(metric, ascending)
        fig, ax1 = plt.subplots(figsize=figsize)
        pos = np.arange(len(array_metric))
        rects = ax1.barh(pos, array_metric,
                         align='center',
                         height=0.5,
                         tick_label=array_labels)
        ax1.set_title(str_header)
        for a, b in zip(pos, array_metric):
            plt.text(0.5, a - 0.05, str(round(b, 3)), color='white')
        #plt.show()
        st.pyplot(fig)



    # Сохранение метрик
clasMetricLogger = MetricLogger()


# Отрисовка ROC-кривой
def draw_roc_curve(y_true, y_score, ax, pos_label=1, average='micro'):
    fpr, tpr, thresholds = roc_curve(y_true, y_score,
                                     pos_label=pos_label)
    roc_auc_value = roc_auc_score(y_true, y_score, average=average)
    #fig1 = plt.figure(figsize=(7, 5))
    #plt.figure()
    lw = 2
    ax.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc_value)
    ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    ax.set_xlim([0.0, 1.0])
    ax.set_xlim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic')
    ax.legend(loc="lower right")
    #st.pyplot(fig1)

def clas_train_model(model_name, model, clasMetricLogger):
    model.fit(clas_X_train, clas_Y_train)
    # Предсказание значений
    Y_pred = model.predict(clas_X_test)
    # Предсказание вероятности класса "1" для roc auc
    Y_pred_proba_temp = model.predict_proba(clas_X_test)
    Y_pred_proba = Y_pred_proba_temp[:, 1]

    precision = precision_score(clas_Y_test.values, Y_pred)
    recall = recall_score(clas_Y_test.values, Y_pred)
    f1 = f1_score(clas_Y_test.values, Y_pred)
    roc_auc = roc_auc_score(clas_Y_test.values, Y_pred_proba)

    clasMetricLogger.add('precision', model_name, precision)
    clasMetricLogger.add('recall', model_name, recall)
    clasMetricLogger.add('f1', model_name, f1)
    clasMetricLogger.add('roc_auc', model_name, roc_auc)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    draw_roc_curve(clas_Y_test.values, Y_pred_proba, ax[0])
    plot_confusion_matrix(model, clas_X_test, clas_Y_test.values, ax=ax[1],
                          display_labels=['0', '1'],
                          cmap=plt.cm.Blues, normalize='true')
    fig.suptitle(model_name)
    st.pyplot(fig)
    #plt.show()





st.subheader('Первые 5 значений обучающей выборки')
st.write(data.head())

st.subheader('Первые 5 значений тестовой выборки')
st.write(test.head())

if st.checkbox('Показать статистические характеристики'):
    st.subheader('Обучающая выборка')
    st.write(data.describe())

    st.subheader('Тестовая выборка')
    st.write(test.describe())

if st.checkbox('Дисбаланс классов для Target'):
    fig, ax = plt.subplots(figsize=(2, 2))
    plt.hist(data['target'])
    st.pyplot(fig)

if st.checkbox('Тепловая карта'):
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(data.corr(),annot=True, fmt='.2f')
    ax.set_title('Обучающая выборка')
    st.pyplot(fig)

if st.checkbox('Логистическая регрессия'):
    clas_train_model('LogR', LogisticRegression(), clasMetricLogger)

if st.checkbox('Медод ближайших соседей'):
    cv_slider = st.slider('Количество фолдов:', min_value=3, max_value=10, value=5, step=1)

    # Вычислим количество возможных ближайших соседей
    rows_in_one_fold = int(data_len / cv_slider)
    allowed_knn = int(rows_in_one_fold * (cv_slider - 1))
    st.write('Количество строк в наборе данных - {}'.format(data_len))
    st.subheader('Метод ближайших соседей')
    st.write('Максимальное допустимое количество ближайших соседей с учетом выбранного количества фолдов - {}'.format(
        allowed_knn))

    cv_knn = st.slider('Количество ближайших соседей:', min_value=1, max_value=allowed_knn, value=5, step=1)


    clas_train_model('KNN', KNeighborsClassifier(n_neighbors=cv_knn), clasMetricLogger)

if st.checkbox('Машина опорных векторов'):
    clas_train_model('SVC', SVC(probability=True), clasMetricLogger)

if st.checkbox('Решающее дерево'):
    clas_train_model('Tree', DecisionTreeClassifier(), clasMetricLogger)

if st.checkbox('Случайный лес'):
    clas_train_model('RF', RandomForestClassifier(), clasMetricLogger)

if st.checkbox('Градиентный бустинг'):
    clas_train_model('GB', GradientBoostingClassifier(), clasMetricLogger)

if st.checkbox('Оценка качества'):
    clas_metrics = clasMetricLogger.df['metric'].unique()
    for metric in clas_metrics:
        clasMetricLogger.plot('Метрика: ' + metric, metric, figsize=(7, 6))

