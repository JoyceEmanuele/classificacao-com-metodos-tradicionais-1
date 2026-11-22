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
from sklearn.linear_model import LogisticRegression
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


pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

grid_lr = GridSearchCV(
    pipeline_lr,
    {'lr__C': [0.01, 0.1, 1, 10]},
    cv=5,
    scoring='f1',
    n_jobs=-1
)

print("\nRegressão Logística")
grid_lr.fit(X_amostra, y_amostra)
best_model_lr = grid_lr.best_estimator_

print("Melhores parâmetros:", grid_lr.best_params_)

y_pred_lr = best_model_lr.predict(X_test_cdc)

print("Acurácia:", accuracy_score(y_test_cdc, y_pred_lr))
print("Precisão:", precision_score(y_test_cdc, y_pred_lr))
print("Revocação:", recall_score(y_test_cdc, y_pred_lr))
print("F1:", f1_score(y_test_cdc, y_pred_lr))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test_cdc, y_pred_lr))
print("\nRelatório de classificação:")
print(classification_report(y_test_cdc, y_pred_lr))



scaler_viz = StandardScaler()
X_amostra_scaled = scaler_viz.fit_transform(X_amostra)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_amostra_scaled)

lr_viz = LogisticRegression(C=grid_lr.best_params_['lr__C'], class_weight='balanced', max_iter=1000, random_state=42)
lr_viz.fit(X_pca, y_amostra)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = lr_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_amostra == 0, 0], X_pca[y_amostra == 0, 1], label='Sem diabetes', alpha=0.3, s=10)
plt.scatter(X_pca[y_amostra == 1, 0], X_pca[y_amostra == 1, 1], label='Com diabetes', alpha=0.3, s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Regressão Logística - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
