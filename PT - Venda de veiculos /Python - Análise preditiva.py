import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Carregar os dados
df = pd.read_csv("mock_vehicle_sales_database.csv")

# Filtrar veículos usados (novos têm preço tabelado)
df_used = df[df['Condition'] == 'Used'].copy()

# Features e Target
features = ['Vehicle_Year', 'Mileage_KM', 'Vehicle_Make', 'Vehicle_Category']
X = df_used[features]
y = df_used['Total_Price_BRL']

# One-Hot Encoding para categorias
X_encoded = pd.get_dummies(X, columns=['Vehicle_Make', 'Vehicle_Category'], drop_first=True)

# Divisão Treino e Teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 2. Treinamento do Modelo de Machine Learning
rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train, y_train)

# Previsões
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)

# Importância das Features
feature_importances = pd.Series(rf_model.feature_importances_, index=X_encoded.columns).sort_values(ascending=False).head(8)

# 3. Construção do Dashboard Visual (Estilo Dark Mode Executivo)
plt.style.use('dark_background')
fig = plt.figure(figsize=(16, 10), facecolor='#121212')
fig.suptitle(f'Painel de Precificação Dinâmica com Machine Learning (R² = {r2:.2f})\nAnálise de Preços de Veículos Usados', 
             fontsize=20, fontweight='bold', color='#E0E0E0', y=0.98)

grid = plt.GridSpec(2, 2, wspace=0.3, hspace=0.35, figure=fig)
color_primary, color_secondary, color_bg = '#00ADB5', '#FF2E63', '#1E1E1E'

# Gráfico 1: Real vs Predito
ax1 = fig.add_subplot(grid[0, 0], facecolor=color_bg)
sns.scatterplot(x=y_test, y=y_pred, alpha=0.7, color=color_primary, s=80, ax=ax1)
min_val, max_val = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
ax1.plot([min_val, max_val], [min_val, max_val], color=color_secondary, linestyle='--', linewidth=2)
ax1.set(title='Precisão do Modelo: Preço Real vs. Previsto', xlabel='Preço Real (BRL)', ylabel='Preço Previsto pelo ML (BRL)')
ax1.grid(color='#333333', linestyle=':', alpha=0.7)

# Gráfico 2: Importância das Variáveis
ax2 = fig.add_subplot(grid[0, 1], facecolor=color_bg)
labels = [idx.replace('Vehicle_Make_', 'Marca: ').replace('Vehicle_Category_', 'Categoria: ') for idx in feature_importances.index]
sns.barplot(x=feature_importances.values, y=labels, palette='viridis', ax=ax2)
ax2.set(title='Fatores que Mais Impactam o Valor do Veículo', xlabel='Importância Relativa')
ax2.grid(axis='x', color='#333333', linestyle=':', alpha=0.7)

# Gráfico 3: Simulação Preditiva (Efeito da Quilometragem)
ax3 = fig.add_subplot(grid[1, :], facecolor=color_bg)
sim_data = pd.DataFrame(columns=X_encoded.columns)
sim_data.loc[0] = 0
sim_data['Vehicle_Year'] = 2018
if 'Vehicle_Category_SUV' in sim_data.columns: sim_data['Vehicle_Category_SUV'] = 1

mileage_range = np.linspace(10000, 150000, 50)
predicted_prices = [rf_model.predict(sim_data.assign(Mileage_KM=km))[0] for km in mileage_range]

ax3.plot(mileage_range, predicted_prices, color=color_secondary, linewidth=3, marker='o')
ax3.fill_between(mileage_range, predicted_prices, min(predicted_prices)*0.9, color=color_secondary, alpha=0.2)
ax3.set(title='Simulação Preditiva: Curva de Depreciação por Quilometragem (SUV 2018)', xlabel='Quilometragem (KM)', ylabel='Valor Estimado (BRL)')
ax3.grid(color='#333333', linestyle=':', alpha=0.7)

plt.tight_layout()
plt.show()
