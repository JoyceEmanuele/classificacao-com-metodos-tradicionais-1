from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


cdc = fetch_openml(data_id=46841, as_frame=True)

X = cdc.data
y = cdc.target.astype(int)

X_train_cdc, X_test_cdc, y_train_cdc, y_test_cdc = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_amostra, _, y_amostra, _ = train_test_split(
    X_train_cdc,
    y_train_cdc,
    train_size=10000,
    random_state=42,
    stratify=y_train_cdc
)

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

grid_knn = GridSearchCV(
    pipeline_knn,
    {'knn__n_neighbors': [3, 5, 7, 11, 15, 21]},
    cv=5,
    scoring='f1',
    n_jobs=-1
)

print("\nKNN")
grid_knn.fit(X_amostra, y_amostra)
best_model_knn = grid_knn.best_estimator_

print("Melhores parâmetros:", grid_knn.best_params_)

y_pred_knn_treino = best_model_knn.predict(X_amostra)

print("Acurácia treino:", accuracy_score(y_amostra, y_pred_knn_treino))
print("Precisão treino:", precision_score(y_amostra, y_pred_knn_treino))
print("Revocação treino:", recall_score(y_amostra, y_pred_knn_treino))
print("F1 treino:", f1_score(y_amostra, y_pred_knn_treino))
print("\nMatriz de confusão treino:")
print(confusion_matrix(y_amostra, y_pred_knn_treino))
print("\nRelatório de classificação treino:")
print(classification_report(y_amostra, y_pred_knn_treino))

y_pred_knn = best_model_knn.predict(X_test_cdc)

print("Acurácia teste:", accuracy_score(y_test_cdc, y_pred_knn))
print("Precisão teste:", precision_score(y_test_cdc, y_pred_knn))
print("Revocação teste:", recall_score(y_test_cdc, y_pred_knn))
print("F1 teste:", f1_score(y_test_cdc, y_pred_knn))
print("\nMatriz de confusão teste:")
print(confusion_matrix(y_test_cdc, y_pred_knn))
print("\nRelatório de classificação teste:")
print(classification_report(y_test_cdc, y_pred_knn))




scaler_viz = StandardScaler()
X_amostra_scaled = scaler_viz.fit_transform(X_amostra)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_amostra_scaled)

knn_viz = KNeighborsClassifier(n_neighbors=grid_knn.best_params_['knn__n_neighbors'])
knn_viz.fit(X_pca, y_amostra)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_amostra == 0, 0], X_pca[y_amostra == 0, 1], label='Sem diabetes', alpha=0.3, s=10)
plt.scatter(X_pca[y_amostra == 1, 0], X_pca[y_amostra == 1, 1], label='Com diabetes', alpha=0.3, s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'KNN (k={grid_knn.best_params_["knn__n_neighbors"]}) - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
