# Classificação de Potabilidade da Água

## Visão Geral
Este projeto treina e avalia modelos de machine learning para prever a potabilidade da água. O experimento é rastreado usando MLflow, e os modelos Random Forest e XGBoost são utilizados para classificação.
Não é do escopo deste projeto a criação de features e o tunning do modelo, rodamos 3 versões simples para utilizarmos o mlflow para logging de parâmetros e métricas importantes do modelo.
Utilizamos o poetry para fazer gerenciamento de dependências e empacotamento para projetos em Python. No arquivo pyproject.toml, temos "tool.poetry.dependencies" que representam as dependencias. Logo abaixo temos a configuração do Ruff. O Ruff é uma ferramenta para desenvolvedores que nos ajuda com linting e formatação de código Python. Cada commit que nós damos só será aceito se estivermos de acordo com o padrao decidido.
Utilizamos o pre-commit para automatizar verificações de código antes de um commit no Git. Ele ajuda a garantir que o código siga padrões de qualidade, estilo e formatação, evitando problemas antes que o código seja enviado para o repositório.
Antes de executar git commit, o pre-commit roda os hooks configurados no repositório.
Se algum hook falhar (exemplo: erro de formatação, espaço em branco desnecessário, erro de lint), o commit é bloqueado até que o problema seja corrigido.
Isso nos obriga a corrigir erros antes de enviar código quebrado ou mal formatado para o repositório.

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

### 3. `model_evaluation`
Módulo que contém funções para avaliar modelos de classificação binária.
- `calculate_metrics(y_true, y_pred, y_pred_proba)`: Calcula métricas de avaliação como acurácia, precisão, recall e F1-score.
- `plot_confusion_matrix(y_true, y_pred)`: Plota a matriz de confusão.
- `plot_precision_recall_curve(y_true, y_pred_proba)`: Plota a curva de precisão-recall.
- `print_classification_report(y_true, y_pred)`: Exibe o relatório de classificação.
- `evaluate_model(y_true, y_pred, y_pred_proba)`: Avalia o modelo e agrega todas as métricas e visualizações relevantes.

## Resultados Esperados
Cada execução gera métricas de avaliação do modelo, que podem ser acessadas no MLflow, permitindo a comparação de desempenho entre os diferentes modelos. Os modelos são salvos dentro do artefato, o que nos leva a poder utilizar o que possui a melhor performance.
No nosso caso o xgboost 2 possuiu melhor performance.
