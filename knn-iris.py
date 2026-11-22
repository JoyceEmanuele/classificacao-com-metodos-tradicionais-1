import pandas as pd
from sklearn.model_selection import train_test_split
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


iris_df = pd.read_csv('Iris.csv')
iris_df = iris_df.drop(columns=['Id'])


X_iris = iris_df.drop(columns=['Species'])
y_iris = pd.Categorical(iris_df['Species']).codes

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris,
    y_iris,
    test_size=0.2,
    random_state=42,
    stratify=y_iris
)

pipeline_knn = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

grid_knn = GridSearchCV(
    pipeline_knn,
    {'knn__n_neighbors': [3, 5, 7, 11, 15, 21]},
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

print("\nKNN")
grid_knn.fit(X_train_iris, y_train_iris)
best_model_knn = grid_knn.best_estimator_

print("Melhores parâmetros:", grid_knn.best_params_)

y_pred_knn_treino = best_model_knn.predict(X_train_iris)

print("Acurácia treino:", accuracy_score(y_train_iris, y_pred_knn_treino))
print("Precisão treino:", precision_score(y_train_iris, y_pred_knn_treino, average='weighted'))
print("Revocação treino:", recall_score(y_train_iris, y_pred_knn_treino, average='weighted'))
print("F1 treino:", f1_score(y_train_iris, y_pred_knn_treino, average='weighted'))
print("\nMatriz de confusão treino:")
print(confusion_matrix(y_train_iris, y_pred_knn_treino))
print("\nRelatório de classificação treino:")
print(classification_report(y_train_iris, y_pred_knn_treino))


y_pred_knn = best_model_knn.predict(X_test_iris)

print("Acurácia teste:", accuracy_score(y_test_iris, y_pred_knn))
print("Precisão teste:", precision_score(y_test_iris, y_pred_knn, average='weighted'))
print("Revocação teste:", recall_score(y_test_iris, y_pred_knn, average='weighted'))
print("F1 teste:", f1_score(y_test_iris, y_pred_knn, average='weighted'))
print("\nMatriz de confusão teste:")
print(confusion_matrix(y_test_iris, y_pred_knn))
print("\nRelatório de classificação teste:")
print(classification_report(y_test_iris, y_pred_knn))



scaler_viz = StandardScaler()
X_train_scaled = scaler_viz.fit_transform(X_train_iris)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

knn_viz = KNeighborsClassifier(n_neighbors=grid_knn.best_params_['knn__n_neighbors'])
knn_viz.fit(X_pca, y_train_iris)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = knn_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_train_iris == 0, 0], X_pca[y_train_iris == 0, 1], label='Setosa', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 1, 0], X_pca[y_train_iris == 1, 1], label='Versicolor', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 2, 0], X_pca[y_train_iris == 2, 1], label='Virginica', alpha=0.6, s=30)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'KNN (k={grid_knn.best_params_["knn__n_neighbors"]}) - Iris - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
