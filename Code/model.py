import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

file = r'C:\Users\admin\Desktop\Stage4eme\Code\CompositionPortefeuille.xls'

df = pd.read_excel(file)

# Describe the data
print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nData Description:")
print(df.describe())

# Exploratory Data Analysis (EDA)
# 1. Distribution of numerical features
df.hist(bins=20, figsize=(15, 8))
plt.suptitle('Distribution of Numerical Features')
plt.show()

# 2. Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=['number']).corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# 3. Pair Plot (Include only numerical columns)
numerical_features = df.select_dtypes(include=['number']).columns
if len(numerical_features) > 0:
    sns.pairplot(df[numerical_features])
    plt.suptitle('Pair Plot of Numerical Features', y=1.02)
    plt.show()

# 4. Count Plot for Categorical Features
# Get categorical features
relevant_categorical_features = [
    'Etat du dossier',
    'Titre de civilité de l\'assuré',
    'Profession de l\'assuré',
    'Ville de l\'assuré'
]

# Number of relevant categorical features
num_features = len(relevant_categorical_features)
num_cols = 2  # Number of columns in the subplot grid
num_rows = (num_features + num_cols - 1) // num_cols  # Compute number of rows needed

# Create a wide figure
plt.figure(figsize=(15, 5 * num_rows))

for i, feature in enumerate(relevant_categorical_features):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.countplot(y=df[feature], order=df[feature].value_counts().index)  # Order by frequency for better visualization
    plt.title(f'Count Plot for {feature}')
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()

# 5. Box Plot for Numerical Features
numerical_features = df.select_dtypes(include=['number']).columns
num_numerical_features = len(numerical_features)
num_cols = 3  # Number of columns in the subplot grid
num_rows = (num_numerical_features + num_cols - 1) // num_cols  # Compute number of rows needed

plt.figure(figsize=(15, 5 * num_rows))

for i, feature in enumerate(numerical_features):
    plt.subplot(num_rows, num_cols, i + 1)
    sns.boxplot(x=df[feature])
    plt.title(f'Box Plot for {feature}')
    plt.xlabel('')
    plt.ylabel('')

plt.tight_layout()
plt.show()

# 7. Missing Values Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

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

# Display the first few rows of the cleaned dataset
print("\nClean Data Head:")
print(df.head())

# Select relevant columns for analysis
relevant_columns = ['PolicyNumber', 'Insured', 'Gender', 'DateOfBirth', 'City', 'SumInsured', 'Premium', 'PremiumFrequency', 'MathematicalProvision', 'Product', 'TotalPremiums', 'Age', 'Duration']
df_relevant = df[relevant_columns]

# Encode categorical variables
le = LabelEncoder()
df_relevant['Gender'] = le.fit_transform(df_relevant['Gender'])
df_relevant['City'] = le.fit_transform(df_relevant['City'])
df_relevant['Product'] = le.fit_transform(df_relevant['Product'])

# Standardize numerical features
scaler = StandardScaler()
df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']] = scaler.fit_transform(df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']])

# Perform Principal Component Analysis (PCA)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration']])
df_relevant['PC1'] = X_pca[:, 0]
df_relevant['PC2'] = X_pca[:, 1]
df_relevant['PC3'] = X_pca[:, 2]

# Perform K-Means Clustering
kmeans = KMeans(n_clusters=5, random_state=42)
df_relevant['Cluster'] = kmeans.fit_predict(df_relevant[['PC1', 'PC2', 'PC3']])

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Cluster', data=df_relevant)
plt.title('K-Means Clustering on Principal Components')
plt.show()

# Analyze cluster characteristics
print("\nCluster Characteristics:")
for cluster in df_relevant['Cluster'].unique():
    cluster_data = df_relevant[df_relevant['Cluster'] == cluster]
    print(f"Cluster {cluster}:")
    print(cluster_data.describe())
    print()