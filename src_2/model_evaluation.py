"""
Funções para avaliar modelos de classificação binária e exibir métricas, matriz de confusão e curva ROC, etc.
"""

from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(y_true, y_pred, y_pred_proba: Optional[list] = None) -> dict:
    """
    Calcula métricas de avaliação para um problema de classificação binária.

    Args:
        y_true: Valores reais (ground truth).
        y_pred: Previsões do modelo (classes).
        y_pred_proba: Probabilidades previstas para a classe positiva (opcional).

    Returns:
        Um dicionário contendo as métricas calculadas.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
    }

    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)

    return metrics


def plot_confusion_matrix(y_true, y_pred, title: str = 'Confusion Matrix'):
    """
    Plota a matriz de confusão para um problema de classificação binária.

    Args:
        y_true: Valores reais (ground truth).
        y_pred: Previsões do modelo (classes).
        title: Título do gráfico.
    """
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.show()


def plot_roc_curve(y_true, y_pred_proba, title: str = 'ROC Curve'):
    """
    Plota a curva ROC para um problema de classificação binária.

    Args:
        y_true: Valores reais (ground truth).
        y_pred_proba: Probabilidades previstas para a classe positiva.
        title: Título do gráfico.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    roc_auc = roc_auc_score(y_true, y_pred_proba)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.show()


def print_classification_report(y_true, y_pred):
    """
    Imprime o relatório de classificação para um problema de classificação binária.

    Args:
        y_true: Valores reais (ground truth).
        y_pred: Previsões do modelo (classes).
    """
    report = classification_report(y_true, y_pred)
    print('Classification Report:')
    print(report)


def evaluate_model(y_true, y_pred, y_pred_proba: Optional[list] = None):
    """
    Avalia um modelo de classificação binária e exibe métricas, matriz de confusão e curva ROC.

    Args:
        y_true: Valores reais (ground truth).
        y_pred: Previsões do modelo (classes).
        y_pred_proba: Probabilidades previstas para a classe positiva (opcional).
    """
    # Calcular métricas
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    print('Métricas Calculadas:')
    for key, value in metrics.items():
        print(f'{key}: {value:.4f}')

    # Plotar matriz de confusão
    plot_confusion_matrix(y_true, y_pred)

    # Plotar curva ROC (se y_pred_proba for fornecido)
    if y_pred_proba is not None:
        plot_roc_curve(y_true, y_pred_proba)

    # Imprimir relatório de classificação
    print_classification_report(y_true, y_pred)
