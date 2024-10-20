from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Carregar o conjunto de dados Iris
iris = datasets.load_iris()
features = iris.data
targets = iris.target

# 2. Calcular a soma das características
features_all = np.sum(features, axis=1).reshape(-1, 1)  # Soma das características para cada observação

# 3. Criar e treinar o modelo de regressão linear
model = LinearRegression()
model.fit(features_all, targets)

# 4. Fazer previsões
predictions = model.predict(features_all)

# 5. Obter coeficientes
coef_angular = model.coef_[0]
coef_linear = model.intercept_

print(f"Coeficiente angular (slope): {coef_angular}")
print(f"Coeficiente linear (intercepto): {coef_linear}")

# 6. Criar o gráfico
plt.figure(figsize=(10, 8))
plt.scatter(features_all, targets, color='red', alpha=1.0, label='Dados Reais')
plt.plot(features_all, predictions, color='blue', label=f'Regressão Linear (y = {coef_angular:.2f}x + {coef_linear:.2f})')

plt.title('Iris Dataset Scatter Plot com Regressão Linear')
plt.xlabel("Soma das Features")
plt.ylabel('Targets')
plt.legend()
plt.grid()
plt.show()

# 7. Fazer uma nova previsão
novo_valor = [[5]]  # Exemplo de soma das features
y_pred = model.predict(novo_valor)
print(f"Para soma das features {novo_valor[0][0]}, o valor de Y é: {y_pred[0]}")
