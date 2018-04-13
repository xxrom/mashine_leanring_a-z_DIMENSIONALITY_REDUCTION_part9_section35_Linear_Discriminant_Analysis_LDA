# LDA Linear Discriminant Analysis supervised model
# supervised = алгоритм зависит от выходного параметра y_train
# accuracy = 100% !!!

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Wine.csv')
X = dataset.iloc[:, 0:13].values
y = dataset.iloc[:, 13].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# убрали варнинг используя эту библиотеку model_selection
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components = 2)  # просто 2 ставим, без анализа (не как в PDA)
# fit_ = understanding the structure X_train and Y_train знать информацию
# о классах, что бы максимально точно разделить данные по классам
X_train = lda.fit_transform(X_train, y_train)
# оставляем только 2 параметра + scale
X_test = lda.transform(X_test) # модифицируем только X_test без анализа
# не анализируем ничего, поэтому не нужно писать fit_, а только transform

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0) # инициализация модели
classifier.fit(X_train, y_train) # закидываем в модель данные для обучения модели

# Predicting the Test set results
y_pred = classifier.predict(X_test) # предсказываем данные из X_test

# Making the Confusion Matrix # узнаем насколько правильная модель
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred) # закиыдваем тестовые и предсказанные данные
# данные в 3х3 матрице [[14 0 0], [0, 16, 0], [0, 0, 6]]
# точность модели 36/36 = 100% !!! на тестовых данных


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train

# подготавливаем матрицу для нашего поля данных с шагом сетки 0.01
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# вся магия тут, раскрашиваем данные по всему полотку X1, X2
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

# границы для областей указываем?
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# все точки рисуем на полотне, которые у нас есть
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Linear Discriminant Analysis (Training set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend() # в правом верхнем углу рисует соотношение точек и из значений
plt.show()


# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test

# подготавливаем матрицу для нашего поля данных с шагом сетки 0.01
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

# вся магия тут, раскрашиваем данные по всему полотку X1, X2
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, cmap = ListedColormap(('red', 'green', 'blue')))

# границы для областей указываем?
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

# все точки рисуем на полотне, которые у нас есть
for i, j in enumerate(np.unique(y_set)):
  plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
              c = ListedColormap(('red', 'green', 'blue'))(i), label = j)
plt.title('Linear Discriminant Analysis (Test set)')
plt.xlabel('LD1')
plt.ylabel('LD2')
plt.legend() # в правом верхнем углу рисует соотношение точек и из значений
plt.show()