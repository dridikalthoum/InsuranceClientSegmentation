import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.ensemble import RandomForestRegressor
import shap
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy import stats
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from scipy.stats import gaussian_kde



# Load dataset
file = r'Code\CompositionPortefeuille.xls'
df = pd.read_excel(file)

# Describe the data
print("Data Head:")
print(df.head())
print("\nData Info:")
print(df.info())
print("\nData Description:")
print(df.describe())

print(df["Profession de l'assuré"].unique())

# Renaming columns
df.rename(columns={
    "Numéro de dossier": "PolicyNumber",
    "Souscripteur": "Subscriber",
    "Assuré": "Insured",
    "Date de naissance de l'assuré": "DateOfBirth",
    "Titre de civilité de l'assuré": "Title",
    "Adresse de l'assuré": "Address",
    "Ville de l'assuré": "City",
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

# Define standardization function
def standardize_occupation(occupation):
    if pd.isna(occupation):
        return 'Unknown'
    occupation = occupation.upper().strip()  # Convert to uppercase and remove leading/trailing spaces
    if 'DIRECTEUR' in occupation:
        return 'Directeur'
    elif 'CADRE DE BANQUE' in occupation:
        return 'Cadre de banque'
    elif 'AGENT GENERAL' in occupation:
        return 'Agent général'
    elif 'ENSEIGNANT' in occupation or 'PROFESSEUR' in occupation:
        return 'Enseignant'
    elif 'SURVEILLANT DE TRAVAUX' in occupation or 'FONCTIONNAIRE' in occupation:
        return 'Fonctionnaire'
    elif 'MEDECIN' in occupation:
        return 'Médecin'
    elif 'RETRAITE' in occupation:
        return 'Retraite'
    else:
        return 'Unknown'

# Apply standardization
df["Profession de l'assuré"] = df["Profession de l'assuré"].apply(standardize_occupation)

# Check the unique values after standardization
print(df["Profession de l'assuré"].unique())

# Rename the column
df.rename(columns={"Profession de l'assuré": "Occupation"}, inplace=True)

# Encode Occupation with ordinal values
occupation_mapping = {
    'Directeur': 1, 
    'Cadre de banque': 1, 
    'Agent général': 1,
    'Médecin': 2, 
    'Enseignant': 3, 
    'Professeur': 3,
    'Surveillant de travaux': 4, 
    'Fonctionnaire': 4,
    'Retraite': 5,
    'Unknown': 0
}

# Replace NaN values with 'Unknown'
df['Occupation'] = df['Occupation'].replace(np.nan, 'Unknown')
df['Occupation'] = df['Occupation'].map(occupation_mapping)

# Select only the numerical columns for correlation matrix calculation
numerical_cols = df.select_dtypes(include=[np.number])

# Calculate the correlation matrix for numerical features
correlation_matrix = numerical_cols.corr()

# Set up the matplotlib figure
plt.figure(figsize=(14, 10))

# Draw the heatmap with improved aesthetics
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    fmt='.2f', # Format annotations to 2 decimal places
    cmap='viridis', # Use a perceptually uniform color map
    linewidths=0.5, 
    linecolor='black', # Line color to separate cells
    cbar_kws={'shrink': .75}, # Adjust color bar size
    annot_kws={"size": 10, "weight": "bold"} # Annotation font size and weight
)

# Improve the layout
plt.title('Correlation Matrix of Numerical Features', fontsize=18, weight='bold')
plt.xticks(rotation=45, ha='right') # Rotate x-axis labels for better readability
plt.yticks(rotation=0) # Keep y-axis labels horizontal

# Display the plot
plt.tight_layout() # Adjust layout to prevent clipping
plt.show()

# Feature engineering
df['Age'] = (df['EffectiveDate'] - df['DateOfBirth']).dt.days // 365
df['Duration'] = (df['ExpirationDate'] - df['EffectiveDate']).dt.days // 365
df['Gender'] = df['Title'].apply(lambda x: 'Male' if x == 'Monsieur' else 'Female')
df['CustomerTenure'] = (df['EffectiveDate'] - df['DateOfBirth']).dt.days // 365
df['AverageClaimAmount'] = df['TotalPremiums'] / (df['Duration'] + 1)  # Avoid division by zero

# Adjust ClaimSeverity and LifetimeValue
df['ClaimSeverity'] = df['SumInsured'] / df['Premium']  # Example calculation, adjust as needed
df['LifetimeValue'] = df['TotalPremiums'] / (df['Duration'] + 1)  # Example calculation, adjust as needed

# Handle missing values with mean imputation
df.fillna(df.mean(numeric_only=True), inplace=True)  # Fill missing numerical values with the mean

# Select relevant columns for analysis
relevant_columns = ['PolicyNumber', 'Insured', 'Gender', 'Age', 'City', 'Occupation', 'SumInsured', 'Premium', 'MathematicalProvision', 
                    'Product', 'TotalPremiums', 'Duration', 'ClaimSeverity', 'LifetimeValue', 'CustomerTenure', 'AverageClaimAmount']
df_relevant = df[relevant_columns].copy()

# Features to check for outliers
features_to_check = ['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration', 
                      'ClaimSeverity', 'LifetimeValue', 'CustomerTenure', 'AverageClaimAmount']

# Create a grid of subplots
n_features = len(features_to_check)
n_cols = 3  # Number of columns for subplots
n_rows = (n_features + n_cols - 1) // n_cols  # Calculate number of rows

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows), constrained_layout=True)

# Flatten axes array for easy iteration
axes = axes.flatten()

# Custom bright colors for boxplots
colors = sns.color_palette("bright", n_colors=len(features_to_check))

# Plot boxplots for each feature
for i, feature in enumerate(features_to_check):
    sns.boxplot(
        data=df_relevant,
        x=feature,
        ax=axes[i],
        color=colors[i]  # Apply bright color directly
    )
    axes[i].set_title(f'Boxplot of {feature}', fontsize=14, weight='bold')
    axes[i].set_xlabel('')
    axes[i].set_ylabel('Value', fontsize=12)
    axes[i].tick_params(axis='both', which='major', labelsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Remove empty subplots if any
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.show()

# Calculate Z-scores for the features
z_scores = np.abs(stats.zscore(df_relevant[features_to_check]))

# Adjusted threshold for outlier removal to avoid excessive data loss
threshold = 2.5  # Increased from 3 to reduce data loss
df_relevant_no_outliers = df_relevant[(z_scores < threshold).all(axis=1)]

# Print the shape of the dataset before and after outlier removal
print(f"Original data shape: {df_relevant.shape}")
print(f"Data shape after outlier removal: {df_relevant_no_outliers.shape}")

# Plot distributions for the newly created features
features_to_plot = ['Age', 'Duration', 'ClaimSeverity', 'LifetimeValue', 'CustomerTenure', 'AverageClaimAmount']

# Set up the matplotlib figure
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))
axes = axes.flatten()

# Custom color palette for histograms
colors = sns.color_palette("Set2", n_colors=len(features_to_plot))

for i, feature in enumerate(features_to_plot):
    # Plot histogram
    axes[i].hist(
        df_relevant[feature], 
        bins=30, 
        color=colors[i], 
        alpha=0.6, 
        density=True, 
        edgecolor='black'
    )
    
    # Compute and plot KDE
    data = df_relevant[feature].dropna()
    kde = gaussian_kde(data)
    x = np.linspace(data.min(), data.max(), 1000)
    axes[i].plot(x, kde(x), color='black', linestyle='--', linewidth=2)

    # Customize titles and labels
    axes[i].set_title(f'Distribution of {feature}', fontsize=14, weight='bold')
    axes[i].set_xlabel(feature, fontsize=12)
    axes[i].set_ylabel('Density', fontsize=12)
    axes[i].tick_params(axis='both', which='major', labelsize=10)
    axes[i].grid(True, linestyle='--', alpha=0.7)  # Add grid lines for better readability

# Improve the layout
plt.tight_layout(pad=4.0)
plt.show()

# Encode categorical variables
le = LabelEncoder()
df_relevant['Gender'] = le.fit_transform(df_relevant['Gender'])
df_relevant['City'] = le.fit_transform(df_relevant['City'])
df_relevant['Product'] = le.fit_transform(df_relevant['Product'])

# Standardize numerical features
scaler = StandardScaler()
df_relevant_scaled = df_relevant.copy()
df_relevant_scaled[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration', 
                    'ClaimSeverity', 'LifetimeValue', 'CustomerTenure', 'AverageClaimAmount']] = scaler.fit_transform(
    df_relevant[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 'Age', 'Duration', 
                 'ClaimSeverity', 'LifetimeValue', 'CustomerTenure', 'AverageClaimAmount']]
)

# Perform Principal Component Analysis (PCA)
pca = PCA(n_components=3)
X_pca = pca.fit_transform(df_relevant_scaled[['SumInsured', 'Premium', 'MathematicalProvision', 'TotalPremiums', 
                                             'Age', 'Duration', 'ClaimSeverity', 'LifetimeValue', 
                                             'CustomerTenure', 'AverageClaimAmount']])
df_relevant_scaled['PC1'] = X_pca[:, 0]
df_relevant_scaled['PC2'] = X_pca[:, 1]
df_relevant_scaled['PC3'] = X_pca[:, 2]


colors = ['#800080', '#FF69B4', '#1E90FF'] 
# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the first three principal components with pastel colors
ax.scatter(df_relevant_scaled['PC1'], df_relevant_scaled['PC2'], df_relevant_scaled['PC3'], 
           alpha=0.7, c=np.random.choice(colors, size=len(df_relevant_scaled)), marker='o')

# Add labels and title
ax.set_title('Exploration of Principal Components: Insights from PCA', fontsize=16)
ax.set_xlabel('Principal Component 1', fontsize=14)
ax.set_ylabel('Principal Component 2', fontsize=14)
ax.set_zlabel('Principal Component 3', fontsize=14)

# Show the plot
plt.show()
"""
# Elbow method for optimal number of clusters
# Calculate SSE for each number of clusters
SSE = []
for cluster in range(1, 10):
    kmeans = KMeans(n_clusters=cluster, init='k-means++', random_state=42)
    kmeans.fit(df_relevant_scaled[['PC1', 'PC2', 'PC3']])
    SSE.append(kmeans.inertia_)

# Plot the SSE values
plt.figure(figsize=(14, 8))
plt.plot(range(1, 10), SSE, marker='o', linestyle='-', color='b', markersize=8, linewidth=2)

# Annotate SSE values
for i, sse in enumerate(SSE):
    plt.text(i + 1, sse, f'{sse:.2f}', fontsize=10, ha='right', va='bottom')

# Add vertical line at the "elbow" point (optional, adjust if needed)
elbow_point = 3  # Example elbow point, adjust according to your plot
plt.axvline(x=elbow_point, color='r', linestyle='--', linewidth=1, label='Optimal Clusters')

# Improve labels and title
plt.xlabel('Number of Clusters', fontsize=14)
plt.ylabel('Inertia', fontsize=14)
plt.title('Elbow Method for Optimal Number of Clusters', fontsize=16, weight='bold')
plt.xticks(range(1, 10))  # Ensure all x-ticks are visible
plt.grid(True, linestyle='--', alpha=0.7)  # Add grid lines
plt.legend()  # Show legend

# Display the plot
plt.tight_layout()
plt.show()

# Perform K-Means Clustering with the chosen number of clusters
kmeans = KMeans(n_clusters=3, random_state=42)
df_relevant_scaled['KMeans_Cluster'] = kmeans.fit_predict(df_relevant_scaled[['PC1', 'PC2', 'PC3']])

# Calculate and print the silhouette score and Davies-Bouldin Index
sil_score = silhouette_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], kmeans.labels_)
db_index = davies_bouldin_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], df_relevant_scaled['KMeans_Cluster'])
inertia = kmeans.inertia_

print(f'K-Means Silhouette Score: {sil_score:.4f}')
print(f'K-Means Davies-Bouldin Index: {db_index:.4f}')
print(f'K-Means Inertia: {inertia:.4f}')

# Visualize the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='KMeans_Cluster', data=df_relevant_scaled, palette='viridis')
plt.title('K-Means Clustering on Principal Components')
plt.show()

# Get the count of each cluster
cluster_counts = df_relevant_scaled['KMeans_Cluster'].value_counts()

# Define cluster labels and colors
cluster_labels = [f'Group{i+1}' for i in range(len(cluster_counts))]
colors = plt.cm.Paired(range(len(cluster_counts)))  # Use a colormap to avoid color issues

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(
    cluster_counts,
    labels=cluster_labels,
    autopct='%1.1f%%',  # Format percentage labels
    startangle=140,
    colors=colors,
    shadow=True,  # Add shadow for 3D effect
    explode=[0.1] * len(cluster_counts)  # Optionally explode all segments equally
)

# Set title and display
plt.title('Distribution of Clusters in KMeans', fontsize=16, weight='bold')
plt.show()

# Perform DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_relevant_scaled['DBSCAN_Cluster'] = dbscan.fit_predict(df_relevant_scaled[['PC1', 'PC2', 'PC3']])

# Calculate and print DBSCAN metrics
try:
    dbscan_sil_score = silhouette_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], df_relevant_scaled['DBSCAN_Cluster'])
except ValueError:
    dbscan_sil_score = 'Not defined (noise points detected)'

print(f'DBSCAN Silhouette Score: {dbscan_sil_score}')

# Perform Hierarchical Clustering
linked = linkage(df_relevant_scaled[['PC1', 'PC2', 'PC3']], method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Determine clusters from Hierarchical Clustering
df_relevant_scaled['Hierarchical_Cluster'] = fcluster(linked, t=3, criterion='maxclust')

# Calculate and print the silhouette score for hierarchical clusters
try:
    hier_sil_score = silhouette_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], df_relevant_scaled['Hierarchical_Cluster'])
except ValueError:
    hier_sil_score = 'Not defined (noise points detected)'

print(f'Hierarchical Clustering Silhouette Score: {hier_sil_score}')

# Visualize the Hierarchical Clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Hierarchical_Cluster', data=df_relevant_scaled, palette='viridis')
plt.title('Hierarchical Clustering on Principal Components')
plt.show()


# Random Forest Regressor Setup
X = df_relevant_scaled.drop(['PolicyNumber', 'Insured', 'Gender', 'City', 'Product', 'KMeans_Cluster', 'DBSCAN_Cluster', 'Hierarchical_Cluster'], axis=1)
y = df_relevant_scaled['Premium']  # Assuming Premium is the target variable

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predictions and feature importance
y_pred = rf.predict(X_test)
feature_importances = rf.feature_importances_

# SHAP Analysis
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(12, 6))
shap.summary_plot(shap_values, X_test)
plt.show()

# Assume df_original contains the original unscaled data
# and df_relevant_scaled contains the scaled data with KMeans clusters

# Add KMeans cluster labels from the scaled data to the original data
df_relevant['KMeans_Cluster'] = df_relevant_scaled['KMeans_Cluster']

# Calculate the average values for original features grouped by KMeans clusters
original_features_summary = df_relevant.groupby('KMeans_Cluster')[['Age', 'Duration', 'Premium', 'SumInsured', 'AverageClaimAmount', 'CustomerTenure', 'ClaimSeverity']].mean()

print("\nOriginal Numerical Features Summary for KMeans Clustering:")
print(original_features_summary)


# Calculate the average values for original features grouped by KMeans clusters
features_summary = df_relevant_scaled.groupby('KMeans_Cluster')[['Gender', 'Occupation', 'City', 'Product']].mean()
print("\nScaled Features Summary for KMeans Clustering:")
print(features_summary)

# Assuming df_relevant_scaled has 'KMeans_Cluster' as well
scaled_features_summary = df_relevant_scaled.groupby('KMeans_Cluster')[['Age', 'Duration', 'Premium', 'SumInsured', 'AverageClaimAmount', 'CustomerTenure', 'ClaimSeverity', 'Gender', 'Occupation', 'Product']].mean()

# Plot average feature values for each KMeans cluster using scaled data
scaled_features_summary.plot(kind='bar', figsize=(15, 8))
plt.title('Average Scaled Feature Values for KMeans Clusters')
plt.xlabel('Cluster')
plt.ylabel('Scaled Value')
plt.legend(loc='best')
plt.show()"""