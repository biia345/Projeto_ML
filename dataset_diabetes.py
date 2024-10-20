import pandas as pd
from sklearn import datasets

# 1. Carregar o conjunto de dados Diabetes
diabetes = datasets.load_diabetes()

# 2. Criar um DataFrame com os dados
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# 3. Adicionar a coluna de targets (progresso da diabetes)
df['progression'] = diabetes.target

# 4. Visualizar os primeiros registros do DataFrame
print(df.head())
