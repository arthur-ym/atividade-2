"""
Classe para treinar um modelo de classificação de potabilidade da água.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


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

    def train_model(self):
        """
        Treina um modelo de RandomForestClassifier.
        """
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print('Modelo treinado com sucesso!')

    def evaluate_model(self):
        """
        Avalia o modelo e retorna as métricas.
        """
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        class_report = classification_report(self.y_test, y_pred)

        print(f'Acurácia: {accuracy}')
        print('Matriz de Confusão:')
        print(conf_matrix)
        print('Relatório de Classificação:')
        print(class_report)

        return accuracy, conf_matrix, class_report

    def log_to_mlflow(self, experiment_name='Water_Potability_Classification'):
        """
        Registra o modelo e as métricas no MLflow.
        """
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log dos parâmetros do modelo
            mlflow.log_param('n_estimators', 100)
            mlflow.log_param('random_state', 42)

            # Log das métricas
            accuracy, conf_matrix, class_report = self.evaluate_model()
            mlflow.log_metric('accuracy', accuracy)
            mlflow.log_text(str(conf_matrix), 'confusion_matrix.txt')
            mlflow.log_text(class_report, 'classification_report.txt')

            # Log do modelo
            mlflow.sklearn.log_model(self.model, 'model')
            print('Modelo e métricas registrados no MLflow!')

    def run(self):
        """
        Executa o pipeline completo: carregar dados, pré-processar, treinar e registrar no MLflow.
        """
        self.load_data()
        self.preprocess_data()
        self.train_model()
        self.log_to_mlflow()
