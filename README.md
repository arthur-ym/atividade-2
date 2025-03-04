# Classificação de Potabilidade da Água

## Visão Geral
Este projeto treina e avalia modelos de machine learning para prever a potabilidade da água. O experimento é rastreado usando MLflow, e os modelos Random Forest e XGBoost são utilizados para classificação.

## Estrutura do Projeto
O projeto segue os seguintes passos:
1. **EDA**: Na pasta EDA, temos uma breve exploração dos dados com comentários sobre as visões geradas.
2. **Definição de classes e funções**: Definição da classe WaterPotabilityClassifier e as funções de mlflowIO e model_evaluation
Após isso, criamos o nosso entrypoint:
3. **Importação de Bibliotecas**: Carregamento dos módulos necessários, incluindo `WaterPotabilityClassifier` e `mlflowIO`.
4. **Treinamento e Avaliação de Modelos**:
   - Random Forest
   - XGBoost (duas variantes)
5. **Registro de Resultados no MLflow**

## Guia de Execução do Entrypoint
### 1. Configuração do Ambiente
Definição do diretório de trabalho para garantir acesso aos módulos corretos:
```python
import os
os.chdir('..')
print(os.getcwd())
```

### 2. Importação de Bibliotecas
```python
from src_2.classifier import WaterPotabilityClassifier
import src_2.mlflowIO as mlflow_io
```

### 3. Execução dos Experimentos
Os experimentos são registrados no MLflow, e os modelos são treinados e avaliados.

#### Experimento 1: Random Forest
```python
mlflow_io.start_run(run_name="random_forest", experiment_name="water_potability")
water_classifier = WaterPotabilityClassifier("EDA/water_potability.csv")
water_classifier.load_data()
water_classifier.preprocess_data()
water_classifier.train_model("random_forest")
water_classifier.run_evaluate_model()
mlflow_io.end_run()
```

#### Experimento 2: XGBoost
```python
mlflow_io.start_run(run_name="xgboost", experiment_name="water_potability")
water_classifier = WaterPotabilityClassifier("EDA/water_potability.csv")
water_classifier.load_data()
water_classifier.preprocess_data()
water_classifier.train_model("xgboost")
water_classifier.run_evaluate_model()
mlflow_io.end_run()
```

#### Experimento 3: XGBoost (Variante 2)
```python
mlflow_io.start_run(run_name="xgboost 2", experiment_name="water_potability")
water_classifier = WaterPotabilityClassifier("EDA/water_potability.csv")
water_classifier.load_data()
water_classifier.preprocess_data()
water_classifier.train_model("xgboost_2")
water_classifier.run_evaluate_model()
mlflow_io.end_run()
```

## Descrição das Classes e Funções
### 1. `WaterPotabilityClassifier`
Classe responsável por carregar os dados, realizar o pré-processamento e treinar diferentes modelos de classificação.

- `__init__(self, data_path)`: Inicializa a classe com o caminho do dataset.
- `load_data(self)`: Carrega os dados do arquivo CSV e trata valores ausentes.
- `preprocess_data(self)`: Divide os dados entre variáveis preditoras (X) e alvo (y), padroniza os dados e separa em conjuntos de treino e teste.
- `train_model(self, model_type)`: Treina um modelo especificado pelo usuário (`random_forest`, `xgboost`, `xgboost_2`) e registra o modelo e seus hiperparâmetros no MLflow.
- `run_evaluate_model(self)`: Avalia o modelo, registra as métricas e salva as figuras no MLflow.

### 2. `mlflowIO`
Módulo responsável pelo rastreamento dos experimentos no MLflow.
- `start_run(run_name, experiment_name)`: Inicia uma execução no MLflow.
- `log_model(model, model_name)`: Registra o modelo treinado.
- `log_params(params)`: Registra os hiperparâmetros do modelo.
- `log_metrics(metrics)`: Registra as métricas de avaliação do modelo.
- `log_figures(figures, folder_name)`: Registra visualizações de análise, como matriz de confusão
