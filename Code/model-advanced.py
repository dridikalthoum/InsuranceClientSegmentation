import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

file = r'C:\Users\admin\Desktop\Stage4eme\Code\CompositionPortefeuille.xls'
df = pd.read_excel(file)

# Renaming columns
df.rename(columns={
    "Numéro de dossier": "PolicyNumber",
    "Souscripteur": "Subscriber",
    "Assuré": "Insured",
    "Date de naissance de l'assuré": "DateOfBirth",
    "Titre de civilité de l'assuré": "Title",
    "Adresse de l'assuré": "Address",
    "Ville de l'assuré": "City",
    "Profession de l'assuré": "Occupation",
    "DateEffet": "EffectiveDate",
    "DateEcheance": "ExpirationDate",
    "Etat du dossier": "PolicyStatus",
    "Capital": "SumInsured",
    "Cotisation": "Premium",
    "Périodicité de cotisation": "PremiumFrequency",
    "Provision mathématique au 23/07/2024": "MathematicalProvision",
    "Produit": "Product",
    "Encaissements": "TotalPremiums"
}, inplace=True)

# Convert date columns to datetime
date_cols = ['DateOfBirth', 'EffectiveDate', 'ExpirationDate']
for col in date_cols:
    df[col] = pd.to_datetime(df[col])

# Feature engineering
df['Age'] = (df['EffectiveDate'] - df['DateOfBirth']).dt.days // 365
df['Duration'] = (df['ExpirationDate'] - df['EffectiveDate']).dt.days // 365
df['Gender'] = df['Title'].apply(lambda x: 'Male' if x == 'Monsieur' else 'Female')

# Handle missing values
df = df.dropna()

# Select relevant columns for analysis
relevant_columns = ['PolicyNumber', 'Insured', 'Gender', 'Age', 'City', 'SumInsured', 'Premium', 'MathematicalProvision', 'Product', 'TotalPremiums', 'Duration']
df_relevant = df[relevant_columns].copy()

# Encode categorical variables
le = LabelEncoder()
df_relevant['Gender'] = le.fit_transform(df_relevant['Gender'])
df_relevant['City'] = le.fit_transform(df_relevant['City'])
df_relevant['Product'] = le.fit_transform(df_relevant['Product'])

# Standardize numerical features
scaler = StandardScaler()
df_relevant_scaled = df_relevant.copy()
df_relevant_scaled[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']] = scaler.fit_transform(df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']])

# Perform Principal Component Analysis (PCA)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df_relevant_scaled[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']])
df_relevant_scaled['PC1'] = X_pca[:, 0]
df_relevant_scaled['PC2'] = X_pca[:, 1]
df_relevant_scaled['PC3'] = X_pca[:, 2]

# Elbow method for optimal number of clusters
SSE = []
for cluster in range(1, 10):
    kmeans = KMeans(n_clusters=cluster, init='k-means++')
    kmeans.fit(df_relevant_scaled[['PC1', 'PC2', 'PC3']])
    SSE.append(kmeans.inertia_)

# Plotting the elbow method results
plt.figure(figsize=(12, 6))
plt.plot(range(1, 10), SSE, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Perform K-Means Clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=4, random_state=42)
df_relevant_scaled['Cluster'] = kmeans.fit_predict(df_relevant_scaled[['PC1', 'PC2', 'PC3']])

# Calculate and print the silhouette score and Davies-Bouldin Index
sil_score = silhouette_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], kmeans.labels_)
db_index = davies_bouldin_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], df_relevant_scaled['Cluster'])
inertia = kmeans.inertia_

print(f'Silhouette Score: {sil_score:.4f}')
print(f'Davies-Bouldin Index: {db_index:.4f}')
print(f'Inertia: {inertia:.4f}')

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_relevant_scaled, palette='viridis')
plt.title('K-Means Clustering on Principal Components')
plt.show()

# Assign clusters to each customer in the original dataset
df_relevant['Cluster'] = df_relevant_scaled['Cluster']

# Compute the average values for each cluster
numeric_columns = ['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']
avg_df = df_relevant.groupby('Cluster')[numeric_columns].mean().reset_index()

# Plot the average values of each feature for each cluster
features_to_plot = ['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']
for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y=feature, data=avg_df, palette='viridis')
    plt.title(f'Average {feature} per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'Average {feature}')
    plt.show()

# Train a RandomForest model to predict premiums
df_model = df_relevant_scaled.drop(['PolicyNumber', 'Insured', 'PC1', 'PC2', 'PC3'], axis=1)
X = df_model.drop('Premium', axis=1)
y = df_model['Premium']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Feature importance using SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize the feature importances
shap.summary_plot(shap_values, X_test, feature_names=X.columns)

# t-SNE Visualization
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(df_relevant_scaled[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']])

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=df_relevant_scaled['Cluster'], palette='viridis')
plt.title('t-SNE Visualization of Clusters')
plt.show()
