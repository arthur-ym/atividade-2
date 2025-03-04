"""
Funções auxiliares para logar parâmetros, métricas, figuras, arquivos e modelos no MLflow.
"""
import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import mlflow


def log_params(params: Dict[str, Any]):
    """
    Loga um dicionário de parâmetros no MLflow.

    Args:
        params (Dict[str, Any]): Dicionário contendo os parâmetros a serem logados.
    """
    for key, value in params.items():
        mlflow.log_param(key, value)
    print(f'Parâmetros logados: {params}')


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    """
    Loga um dicionário de métricas no MLflow.

    Args:
        metrics (Dict[str, float]): Dicionário contendo as métricas a serem logadas.
        step (Optional[int]): Passo (epoch, iteração, etc.) associado às métricas.
    """
    for key, value in metrics.items():
        mlflow.log_metric(key, value, step=step)
    print(f'Métricas logadas: {metrics}')


def log_figures(figures: Dict[str, plt.Figure], artifact_path: str = 'figures'):
    """
    Loga figuras matplotlib no MLflow como artefatos.

    Args:
        figures (Dict[str, plt.Figure]): Dicionário contendo as figuras a serem salvas.
        artifact_path (str): Caminho relativo no MLflow para salvar as figuras.
    """
    for name, fig in figures.items():
        file_path = artifact_path + '/' + f'{name}.png'
        mlflow.log_figure(fig, file_path)
        print(f"Figura '{name}' salva em '{file_path}' e logada no MLflow.")


def log_artifacts(artifacts: Dict[str, str], artifact_path: str = 'artifacts'):
    """
    Loga arquivos como artefatos no MLflow.

    Args:
        artifacts (Dict[str, str]): Dicionário contendo os caminhos dos arquivos a serem logados.
        artifact_path (str): Caminho relativo no MLflow para salvar os artefatos.
    """
    os.makedirs(artifact_path, exist_ok=True)
    for name, file_path in artifacts.items():
        mlflow.log_artifact(file_path, artifact_path)
        print(f"Arquivo '{name}' logado no MLflow em '{artifact_path}'.")


def log_model(model, model_name: str, artifact_path: str = 'model'):
    """
    Loga um modelo no MLflow.

    Args:
        model: Modelo treinado (deve ser compatível com mlflow.sklearn, mlflow.tensorflow, etc.).
        model_name (str): Nome do modelo.
        artifact_path (str): Caminho relativo no MLflow para salvar o modelo.
    """
    mlflow.sklearn.log_model(model, artifact_path)
    print(f"Modelo '{model_name}' logado no MLflow em '{artifact_path}'.")


def start_run(run_name: Optional[str] = None, experiment_name: Optional[str] = None, run_id: Optional[str] = None):
    """
    Inicia ou continua uma run no MLflow.

    Args:
        run_name (Optional[str]): Nome da run.
        experiment_name (Optional[str]): Nome do experimento.
        run_id (Optional[str]): ID da run existente para continuar.
    """
    mlflow.set_tracking_uri('http://localhost:5001/')

    if experiment_name:
        mlflow.set_experiment(experiment_name)

    if run_id:
        mlflow.start_run(run_id=run_id)
        print(f'Run continuada: {run_id} (Experimento: {experiment_name})')
    else:
        mlflow.start_run(run_name=run_name)
        print(f'Nova run iniciada: {run_name} (Experimento: {experiment_name})')


def end_run():
    """
    Finaliza a run atual no MLflow.
    """
    mlflow.end_run()
    print('Run finalizada.')
