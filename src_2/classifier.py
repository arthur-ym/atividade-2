"""
Classe para treinar um modelo de classificação de potabilidade da água.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src_2.mlflowIO import log_figures, log_metrics
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

        Args:
            model_type (str): Tipo do modelo a ser treinado. Pode ser:
                - "random_forest" para RandomForestClassifier
                - "xgboost" para XGBClassifier
                - "catboost" para CatBoostClassifier
        """
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_type == 'xgboost':
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        elif model_type == 'xgboost_2':
            default_xgb_params = {
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
            self.model = XGBClassifier(**default_xgb_params)
        else:
            raise ValueError("Modelo inválido! Escolha entre 'random_forest', 'xgboost' ou 'xgboost_2'.")

        self.model.fit(self.X_train, self.y_train)
        print(f'Modelo {model_type} treinado com sucesso!')

    def run_evaluate_model(self):
        """
        Avalia o modelo e retorna as métricas.
        """
        y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = self.model.predict(self.X_test)
        metrics, confusion_matrix, pr_curve = evaluate_model(self.y_test, y_pred, y_pred_proba)
        log_metrics(metrics)
        log_figures({'pr_curve': pr_curve, 'confusion_matrix': confusion_matrix}, 'model_evaluation')

        return metrics, confusion_matrix, pr_curve

    def log_to_mlflow(self, experiment_name='Water_Potability_Classification'):
        """
        Registra o modelo e as métricas no MLflow.
        """
        mlflow.set_experiment(experiment_name)

        # Log dos parâmetros do modelo
        mlflow.log_param('n_estimators', 100)
        mlflow.log_param('random_state', 42)

        accuracy, conf_matrix, class_report = self.evaluate_model()
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_text(str(conf_matrix), 'confusion_matrix.txt')
        mlflow.log_text(class_report, 'classification_report.txt')

        mlflow.sklearn.log_model(self.model, 'model')
