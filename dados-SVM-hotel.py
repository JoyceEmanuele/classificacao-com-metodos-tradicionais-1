import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.decomposition import PCA

df = pd.read_csv('hotel_bookings.csv')

df = df.drop(columns=['reservation_status', 'reservation_status_date', 'country'])

df['children'] = df['children'].fillna(0)
df['agent'] = df['agent'].fillna(0)
df['company'] = df['company'].fillna(0)

print("estatisticas descritivas")
print(df.describe())

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=['is_canceled'])
y = df['is_canceled']

X_train_hotel, X_test_hotel, y_train_hotel, y_test_hotel = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_amostra, _, y_amostra, _ = train_test_split(
    X_train_hotel,
    y_train_hotel,
    train_size=10000,
    random_state=42,
    stratify=y_train_hotel
)


pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC(class_weight='balanced'))
])


grid_linear = GridSearchCV(
    pipeline,
    {
        'svc__kernel': ['linear'],
        'svc__C': [0.001,0.1, 1, 10]},
        cv=5, scoring='f1',
        n_jobs=-1
)
grid_gausiano = GridSearchCV(
    pipeline,
    {'svc__kernel': ['rbf'],
        'svc__C': [0.1, 1, 10],
        'svc__gamma': [0.01, 0.1, 1]},
        cv=5, scoring='f1',
        n_jobs=-1
)


grid_linear.fit(X_amostra, y_amostra)
best_model_linear = grid_linear.best_estimator_


print("Linear")
print("Melhores parâmetros:", grid_linear.best_params_)

y_pred_linear = best_model_linear.predict(X_test_hotel)

print("Acurácia:", accuracy_score(y_test_hotel, y_pred_linear))
print("Precisão:", precision_score(y_test_hotel, y_pred_linear))
print("Revocação:", recall_score(y_test_hotel, y_pred_linear))
print("F1:", f1_score(y_test_hotel, y_pred_linear))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test_hotel, y_pred_linear))
print("\nRelatório de classificação:")
print(classification_report(y_test_hotel, y_pred_linear))


print("Gausiano")

grid_gausiano.fit(X_amostra, y_amostra)
best_model_gausiano = grid_gausiano.best_estimator_

print("Melhores parâmetros:", grid_gausiano.best_params_)

y_pred_gausiano = best_model_gausiano.predict(X_test_hotel)

print("Acurácia:", accuracy_score(y_test_hotel, y_pred_gausiano))
print("Precisão:", precision_score(y_test_hotel, y_pred_gausiano))
print("Revocação:", recall_score(y_test_hotel, y_pred_gausiano))
print("F1:", f1_score(y_test_hotel, y_pred_gausiano))
print("\nMatriz de confusão:")
print(confusion_matrix(y_test_hotel, y_pred_gausiano))
print("\nRelatório de classificação:")
print(classification_report(y_test_hotel, y_pred_gausiano))



scaler_pca = StandardScaler()
X_amostra_scaled = scaler_pca.fit_transform(X_amostra)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_amostra_scaled)

svm_linear_viz = SVC(kernel='linear', C=grid_linear.best_params_['svc__C'], class_weight='balanced')
svm_linear_viz.fit(X_pca, y_amostra)

x_min, x_max = X_pca[:, 0].min() - 0.5, X_pca[:, 0].max() + 0.5
y_min, y_max = X_pca[:, 1].min() - 0.5, X_pca[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400), np.linspace(y_min, y_max, 400))
Z = svm_linear_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_amostra == 0, 0], X_pca[y_amostra == 0, 1], label='Não cancelado', alpha=0.3, s=10)
plt.scatter(X_pca[y_amostra == 1, 0], X_pca[y_amostra == 1, 1], label='Cancelado', alpha=0.3, s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'SVM Linear (C={grid_linear.best_params_["svc__C"]}) - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()


svm_rbf_viz = SVC(kernel='rbf', C=grid_gausiano.best_params_['svc__C'], gamma=grid_gausiano.best_params_['svc__gamma'], class_weight='balanced')
svm_rbf_viz.fit(X_pca, y_amostra)

Z = svm_rbf_viz.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

plt.figure(figsize=(10, 7))
plt.contourf(xx, yy, Z, alpha=0.25, cmap='RdBu')
plt.contour(xx, yy, Z, colors='k', linewidths=0.8)
plt.scatter(X_pca[y_amostra == 0, 0], X_pca[y_amostra == 0, 1], label='Não cancelado', alpha=0.3, s=10)
plt.scatter(X_pca[y_amostra == 1, 0], X_pca[y_amostra == 1, 1], label='Cancelado', alpha=0.3, s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f'SVM Gaussiano (C={grid_gausiano.best_params_["svc__C"]}, γ={grid_gausiano.best_params_["svc__gamma"]}) - Fronteira de decisão (PCA 2D)')
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
