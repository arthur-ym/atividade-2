"""
Funções para avaliar modelos de classificação binária e exibir métricas, matriz de confusão e curva ROC, etc.
"""

from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
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
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'pr_auc': pr_auc,
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

    Returns:
        figure: Objeto da figura matplotlib.
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], ax=ax)

    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    plt.show()

    return fig


def plot_precision_recall_curve(y_true, y_pred_proba, title: str = 'Precision-Recall Curve'):
    """
    Plota a curva de Precision-Recall.

    Args:
        y_true: Valores reais (ground truth).
        y_pred_proba: Probabilidades previstas para a classe positiva.
        title: Título do gráfico.

    Returns:
        figure: Objeto da figura matplotlib.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)
    ax.legend(loc='lower left')
    ax.grid(True)

    plt.show()

    return fig


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


def evaluate_model(y_true, y_pred, y_pred_proba):
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
    confusion_matrix = plot_confusion_matrix(y_true, y_pred)

    # Plotar curva ROC (se y_pred_proba for fornecido)
    if y_pred_proba is not None:
        pr_curve = plot_precision_recall_curve(y_true, y_pred_proba)

    # Imprimir relatório de classificação
    print_classification_report(y_true, y_pred)
    return metrics, confusion_matrix, pr_curve
