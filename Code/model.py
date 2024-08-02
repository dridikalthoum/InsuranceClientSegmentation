import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

file = r'C:\Users\admin\Desktop\Stage4eme\Code\CompositionPortefeuille.xls'

df = pd.read_excel(file)

# Describe the data
print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nData Description:")
print(df.describe())

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

# Convert 'Title' to 'Gender'
df['Gender'] = df['Title'].apply(lambda x: 'Male' if x == 'Monsieur' else 'Female')

# Handle missing values
print("\nMissing Values:")
print(df.isnull().sum())
df = df.dropna()

# Select relevant columns for analysis
relevant_columns = ['PolicyNumber', 'Insured', 'Gender', 'DateOfBirth', 'City', 'SumInsured', 'Premium', 'PremiumFrequency', 'MathematicalProvision', 'Product', 'TotalPremiums', 'Age', 'Duration']
df_relevant = df[relevant_columns].copy()

# Detecting and handling outliers
q1 = df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']].quantile(0.25)
q3 = df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']].quantile(0.75)
iqr = q3 - q1
outlier_mask = ((df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']] < (q1 - 1.5 * iqr)) | (df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']] > (q3 + 1.5 * iqr))).any(axis=1)
outliers = df_relevant.loc[outlier_mask]

print("Outliers dropped:")
print(outliers)

# Remove outliers
df_relevant = df_relevant[~outlier_mask]

# Display the first few rows of the cleaned dataset
print("\nClean Data Head:")
print(df_relevant.head())

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
frame = pd.DataFrame({'Cluster': range(1, 10), 'SSE': SSE})
plt.figure(figsize=(12, 6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.show()

# Perform K-Means Clustering with the chosen number of clusters (in our case, 4 is the ideal number)
kmeans = KMeans(n_clusters=4, random_state=42)
df_relevant_scaled['Cluster'] = kmeans.fit_predict(df_relevant_scaled[['PC1', 'PC2', 'PC3']])

# Calculate and print the silhouette score
sil_score = silhouette_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], kmeans.labels_, metric='euclidean')
print(f'Silhouette Score: {sil_score}')

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_relevant_scaled, palette='viridis')
plt.title('K-Means Clustering on Principal Components')
plt.show()

# Analyze cluster characteristics
print("\nCluster Characteristics:")
for cluster in df_relevant_scaled['Cluster'].unique():
    cluster_data = df_relevant[df_relevant_scaled['Cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_data.describe())
    print()

# Assign clusters to each customer in the original dataset
df_relevant['Cluster'] = kmeans.predict(df_relevant_scaled[['PC1', 'PC2', 'PC3']])

# Compute the average values for each cluster
# Select only numerical columns for aggregation
numeric_columns = ['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']
avg_df = df_relevant.groupby('Cluster')[numeric_columns].mean().reset_index()

# List of features to plot
features_to_plot = ['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']

# Plot the average values of each feature for each cluster
for feature in features_to_plot:
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y=feature, data=avg_df, palette='viridis')
    plt.title(f'Average {feature} per Cluster')
    plt.xlabel('Cluster')
    plt.ylabel(f'Average {feature}')
    plt.show()
