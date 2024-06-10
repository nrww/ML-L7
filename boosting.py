from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:

    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.3,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params

        self.n_estimators: int = n_estimators

        self.models: list = []
        self.gammas: list = []

        self.learning_rate: float = learning_rate
        self.subsample: float = subsample

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

        self.plot: bool = plot

        self.history = defaultdict(list)

        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

    def fit_new_base_model(self, x, y, predictions):
        """
        Обучает новую базовую модель и добавляет ее в ансамбль.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.
        y : array-like, форма (n_samples,)
            Массив целевых значений.
        predictions : array-like, форма (n_samples,)
            Предсказания текущего ансамбля.

        Примечания
        ----------
        Эта функция добавляет новую модель и обновляет ансамбль.
        """
        
        ind = np.random.choice(np.arange(x.shape[0]), size=int(self.subsample * x.shape[0]))
        x_sub = x[ind]
        residuals_sub = -self.loss_derivative(y[ind], predictions[ind])
        
        model = self.base_model_class(**self.base_model_params)
        model.fit(x_sub, residuals_sub)
        
        new_pred = model.predict(x)
        gamma = self.find_optimal_gamma(y, predictions, self.learning_rate * new_pred)
        
        self.gammas.append(gamma)
        self.models.append(model)
        
        return new_pred, gamma

    def fit(self, x_train, y_train, x_valid, y_valid):
        """
        Обучает модель на тренировочном наборе данных и выполняет валидацию на валидационном наборе.

        Параметры
        ----------
        x_train : array-like, форма (n_samples, n_features)
            Массив признаков для тренировочного набора.
        y_train : array-like, форма (n_samples,)
            Массив целевых значений для тренировочного набора.
        x_valid : array-like, форма (n_samples, n_features)
            Массив признаков для валидационного набора.
        y_valid : array-like, форма (n_samples,)
            Массив целевых значений для валидационного набора.
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        
        for i in range(self.n_estimators):
            
            new_pred, gamma = self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.learning_rate * gamma * new_pred
            
            new_valid_pred = self.models[-1].predict(x_valid)
            valid_predictions += self.learning_rate * gamma * new_valid_pred
            
            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)

            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)
            
            if self.early_stopping_rounds is not None:
                self.validation_loss[i%self.early_stopping_rounds] = valid_loss
                if np.all(valid_loss >= self.validation_loss):
                    break

        if self.plot:
            plt.plot(self.history['train_loss'], label='train')
            plt.plot(self.history['valid_loss'], label='valid')
            plt.legend()
            plt.xlabel('Iteration')
            plt.ylabel('Loss')
            plt.show()
            

    def predict_proba(self, x):
        """
        Вычисляет вероятности принадлежности классу для каждого образца.

        Параметры
        ----------
        x : array-like, форма (n_samples, n_features)
            Массив признаков для набора данных.

        Возвращает
        ----------
        probabilities : array-like, форма (n_samples, n_classes)
            Вероятности для каждого класса.
        """
        pred = np.zeros(x.shape[0])

        for gamma, model in zip(self.gammas, self.models):
            pred += self.learning_rate * gamma * model.predict(x)
        prob = self.sigmoid(pred)
        return np.vstack((1 - prob, prob)).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        """
        Находит оптимальное значение гаммы для минимизации функции потерь.

        Параметры
        ----------
        y : array-like, форма (n_samples,)
            Целевые значения.
        old_predictions : array-like, форма (n_samples,)
            Предыдущие предсказания ансамбля.
        new_predictions : array-like, форма (n_samples,)
            Новые предсказания базовой модели.

        Возвращает
        ----------
        gamma : float
            Оптимальное значение гаммы.

        Примечания
        ----------
        Значение гаммы определяется путем минимизации функции потерь.
        """
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        """
        Возвращает важность признаков в обученной модели.

        Возвращает
        ----------
        importances : array-like, форма (n_features,)
            Важность каждого признака.

        Примечания
        ----------
        Важность признаков определяется по вкладу каждого признака в финальную модель.
        """
        importances = np.zeros_like(self.models[0].feature_importances_)
        for model in self.models:
            importances += model.feature_importances_
        return importances / importances.sum()
