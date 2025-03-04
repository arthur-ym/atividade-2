"""
Classe para treinar um modelo de classificação de potabilidade da água.
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src_2.mlflowIO import log_figures, log_metrics, log_model, log_params
from src_2.model_evaluation import evaluate_model


class WaterPotabilityClassifier:
    """
    Classe para treinar um modelo de classificação de potabilidade da água.
    """

    def __init__(self, data_path):
        """
        Inicializa a classe com o caminho do dataset.
        """
        self.data_path = data_path
        self.df = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Carrega o dataset e realiza o pré-processamento básico.
        """
        self.df = pd.read_csv(self.data_path)
        print('Dados carregados com sucesso!')
        print(self.df.head())

        # Verificar e tratar valores nulos
        print('Valores nulos por coluna:')
        print(self.df.isnull().sum())
        self.df.fillna(self.df.mean(), inplace=True)
        print('Valores nulos após tratamento:')
        print(self.df.isnull().sum())

    def preprocess_data(self):
        """
        Divide os dados em features (X) e target (y), e realiza a padronização.
        """
        X = self.df.drop('Potability', axis=1)
        y = self.df['Potability']

        # Dividir os dados em treino e teste
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Padronizar as features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        print('Dados pré-processados e padronizados!')

    def train_model(self, model_type: str = 'random_forest'):
        """
        Treina um modelo de classificação baseado na escolha do usuário.
        Loga parâmetros do modelo e modelo no MLflow.
        Args:
            model_type (str): Tipo do modelo a ser treinado. Pode ser:
                - "random_forest" para RandomForestClassifier
                - "xgboost" para XGBClassifier
                - "xgboost_2" para XGBClassifier com outros hiperparametros
        """
        if model_type == 'random_forest':
            default_params = {'n_estimators': 100, 'random_state': 42}
            self.model = RandomForestClassifier(**default_params)
        elif model_type == 'xgboost':
            default_params = {'use_label_encoder': False, 'random_state': 42, 'eval_metric': 'logloss'}
            self.model = XGBClassifier(**default_params)
        elif model_type == 'xgboost_2':
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'random_state': 42,
            }
            self.model = XGBClassifier(**default_params)
        else:
            raise ValueError("Modelo inválido! Escolha entre 'random_forest', 'xgboost' ou 'xgboost_2'.")

        self.model.fit(self.X_train, self.y_train)
        print(f'Modelo {model_type} treinado com sucesso!')
        log_model(self.model, model_type)
        log_params(default_params)

    def run_evaluate_model(self):
        """
        Avalia o modelo e retorna as métricas.
        Loga metricas no mlflow.
        """
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = self.model.predict(self.X_test)
        metrics, confusion_matrix, pr_curve = evaluate_model(self.y_test, y_pred, y_pred_proba)
        log_metrics(metrics)
        log_figures({'pr_curve': pr_curve, 'confusion_matrix': confusion_matrix}, 'model_evaluation')

        return metrics, confusion_matrix, pr_curve
