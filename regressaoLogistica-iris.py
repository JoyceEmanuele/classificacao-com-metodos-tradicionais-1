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
from sklearn.linear_model import LogisticRegression
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


pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

grid_lr = GridSearchCV(
    pipeline_lr,
    {'lr__C': [0.01, 0.1, 1, 10]},
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

print("\nRegressão Logística")
grid_lr.fit(X_train_iris, y_train_iris)
best_model_lr = grid_lr.best_estimator_

print("Melhores parâmetros:", grid_lr.best_params_)

y_pred_lr = best_model_lr.predict(X_test_iris)

print("Acurácia:", accuracy_score(y_test_iris, y_pred_lr))
print("Precisão:", precision_score(y_test_iris, y_pred_lr, average='weighted'))
print("Revocação:", recall_score(y_test_iris, y_pred_lr, average='weighted'))
print("F1:", f1_score(y_test_iris, y_pred_lr, average='weighted'))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test_iris, y_pred_lr))
print("\nRelatório de classificação:")
print(classification_report(y_test_iris, y_pred_lr))



scaler_viz = StandardScaler()
X_train_scaled = scaler_viz.fit_transform(X_train_iris)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train_scaled)

lr_viz = LogisticRegression(C=grid_lr.best_params_['lr__C'], class_weight='balanced', max_iter=1000, random_state=42)
lr_viz.fit(X_pca, y_train_iris)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = lr_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdYlBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_train_iris == 0, 0], X_pca[y_train_iris == 0, 1], label='Setosa', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 1, 0], X_pca[y_train_iris == 1, 1], label='Versicolor', alpha=0.6, s=30)
plt.scatter(X_pca[y_train_iris == 2, 0], X_pca[y_train_iris == 2, 1], label='Virginica', alpha=0.6, s=30)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'Regressão Logística (C={grid_lr.best_params_["lr__C"]}) - Iris - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
