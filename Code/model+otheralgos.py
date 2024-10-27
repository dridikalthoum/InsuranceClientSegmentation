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
    fmt='.2f', 
    cmap='viridis', 
    linewidths=0.5, 
    linecolor='black', 
    cbar_kws={'shrink': .75}, 
    annot_kws={"size": 10, "weight": "bold"}
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
df['AverageClaimAmount'] = df['TotalPremiums'] / (df['Duration'] + 1)  

# Adjust ClaimSeverity and LifetimeValue
df['ClaimSeverity'] = df['SumInsured'] / df['Premium']  
df['LifetimeValue'] = df['TotalPremiums'] / (df['Duration'] + 1)  

# Handle missing values with mean imputation
df.fillna(df.mean(numeric_only=True), inplace=True)  

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
    axes[i].grid(True, linestyle='--', alpha=0.7)  

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
colors = plt.cm.Paired(range(len(cluster_counts)))  

# Create a pie chart
plt.figure(figsize=(10, 8))
plt.pie(
    cluster_counts,
    labels=cluster_labels,
    autopct='%1.1f%%',
    startangle=140,
    colors=colors,
    shadow=True,  
    explode=[0.1] * len(cluster_counts) 
)

# Set title and display
plt.title('Distribution of Clusters in KMeans', fontsize=16, weight='bold')
plt.show()

# Perform DBSCAN Clustering
dbscan = DBSCAN(eps=0.5, min_samples=5)
df_relevant_scaled['DBSCAN_Cluster'] = dbscan.fit_predict(df_relevant_scaled[['PC1', 'PC2', 'PC3']])

# Calculate and print DBSCAN metrics
try:
    # Calculate Silhouette Score (only for clusters with labels other than -1)
    dbscan_sil_score = silhouette_score(df_relevant_scaled[df_relevant_scaled['DBSCAN_Cluster'] != -1][['PC1', 'PC2', 'PC3']],
                                        df_relevant_scaled[df_relevant_scaled['DBSCAN_Cluster'] != -1]['DBSCAN_Cluster'])
except ValueError:
    dbscan_sil_score = 'Not defined (noise points detected)'

# Calculate Davies-Bouldin Index
try:
    dbscan_db_score = davies_bouldin_score(df_relevant_scaled[df_relevant_scaled['DBSCAN_Cluster'] != -1][['PC1', 'PC2', 'PC3']],
                                           df_relevant_scaled[df_relevant_scaled['DBSCAN_Cluster'] != -1]['DBSCAN_Cluster'])
except ValueError:
    dbscan_db_score = 'Not defined (noise points detected)'

# Calculate "inertia" as the sum of squared distances from each point to its assigned cluster centroid
def calculate_inertia(data, labels):
    unique_labels = np.unique(labels)
    inertia = 0
    for label in unique_labels:
        if label == -1:  # Skip noise points
            continue
        cluster_points = data[labels == label]
        centroid = cluster_points.mean(axis=0)
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

dbscan_inertia = calculate_inertia(df_relevant_scaled[['PC1', 'PC2', 'PC3']].values, df_relevant_scaled['DBSCAN_Cluster'].values)

# Print the results
print(f'DBSCAN Silhouette Score: {dbscan_sil_score}')
print(f'DBSCAN Davies-Bouldin Index: {dbscan_db_score}')
print(f'DBSCAN Inertia (Sum of Squared Distances to Centroid): {dbscan_inertia}')

# Set the style
plt.style.use('ggplot')

# Create a 2D density plot
plt.figure(figsize=(10, 7))

# Create a density plot using seaborn
sns.kdeplot(
    data=df_relevant_scaled, 
    x='PC1', 
    y='PC2', 
    cmap='Blues',  
    fill=True,    
    alpha=0.5,     
    thresh=0.05    
)

# Overlay the clusters
clusters = df_relevant_scaled['DBSCAN_Cluster']
sns.scatterplot(
    data=df_relevant_scaled,
    x='PC1',
    y='PC2',
    hue=clusters,
    palette='Set2',
    edgecolor='k',  
    alpha=0.7,
    s=50,  
    legend='full'
)

# Add plot labels and title
plt.title('DBSCAN Clustering Density Plot (PC1 vs. PC2)', fontsize=15)
plt.xlabel('PC1', fontsize=12)
plt.ylabel('PC2', fontsize=12)

# Show plot
plt.tight_layout()
plt.show()


# Perform Hierarchical Clustering
linked = linkage(df_relevant_scaled[['PC1', 'PC2', 'PC3']], method='ward')
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.show()

# Determine clusters from Hierarchical Clustering
df_relevant_scaled['Hierarchical_Cluster'] = fcluster(linked, t=3, criterion='maxclust')

# Calculate Silhouette Score
try:
    hier_sil_score = silhouette_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], df_relevant_scaled['Hierarchical_Cluster'])
except ValueError:
    hier_sil_score = 'Not defined (single cluster detected)'

# Calculate Davies-Bouldin Index
try:
    hier_db_score = davies_bouldin_score(df_relevant_scaled[['PC1', 'PC2', 'PC3']], df_relevant_scaled['Hierarchical_Cluster'])
except ValueError:
    hier_db_score = 'Not defined (single cluster detected)'

# Calculate "inertia" as the sum of squared distances from each point to its assigned cluster centroid
def calculate_inertia(data, labels):
    unique_labels = np.unique(labels)
    inertia = 0
    for label in unique_labels:
        cluster_points = data[labels == label]
        centroid = cluster_points.mean(axis=0)
        inertia += np.sum((cluster_points - centroid) ** 2)
    return inertia

hier_inertia = calculate_inertia(df_relevant_scaled[['PC1', 'PC2', 'PC3']].values, df_relevant_scaled['Hierarchical_Cluster'].values)

# Print the results
print(f'Hierarchical Clustering Silhouette Score: {hier_sil_score}')
print(f'Hierarchical Clustering Davies-Bouldin Index: {hier_db_score}')
print(f'Hierarchical Clustering Inertia (Sum of Squared Distances to Centroid): {hier_inertia}')

# Visualize the Hierarchical Clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(x='PC1', y='PC2', hue='Hierarchical_Cluster', data=df_relevant_scaled, palette='viridis')
plt.title('Hierarchical Clustering on Principal Components')
plt.show()


# Random Forest Regressor Setup
X = df_relevant_scaled.drop(['PolicyNumber', 'Insured', 'Gender', 'City', 'Product', 'KMeans_Cluster', 'DBSCAN_Cluster', 'Hierarchical_Cluster'], axis=1)
y = df_relevant_scaled['Premium']  

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
plt.show()

# Sample data for the clusters
clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']
avg_premium = [155.90, 553.85, 7516.25]
avg_claim_amount = [13.38, 42.51, 530.06]
claim_severity = [206.32, 63.87, 17.03]
avg_age = [35.6, 54.7, 53.6]
customer_tenure = [35.6, 54.7, 14.07]
sum_insured = [16045.62, 14187.17, 142214.29]  

# Create subplots for visual representation
fig, ax = plt.subplots(3, 2, figsize=(12, 15))  

# Bar graph for average premium
ax[0, 0].bar(clusters, avg_premium, color='lightblue', edgecolor='black')
ax[0, 0].set_title('Average Premium by Cluster', fontsize=12)
ax[0, 0].set_ylabel('Average Premium', fontsize=10)

# Bar graph for average claim amount
ax[0, 1].bar(clusters, avg_claim_amount, color='lightcoral', edgecolor='black')
ax[0, 1].set_title('Average Claim Amount by Cluster', fontsize=12)
ax[0, 1].set_ylabel('Average Claim Amount', fontsize=10)

# Bar graph for claim severity
ax[1, 0].bar(clusters, claim_severity, color='lightgreen', edgecolor='black')
ax[1, 0].set_title('Claim Severity by Cluster', fontsize=12)
ax[1, 0].set_ylabel('Claim Severity', fontsize=10)

# Bar graph for average age
ax[1, 1].bar(clusters, avg_age, color='lavender', edgecolor='black')
ax[1, 1].set_title('Average Age by Cluster', fontsize=12)
ax[1, 1].set_ylabel('Average Age', fontsize=10)

# Bar graph for customer tenure
ax[2, 0].bar(clusters, customer_tenure, color='peachpuff', edgecolor='black')
ax[2, 0].set_title('Customer Tenure by Cluster', fontsize=12)
ax[2, 0].set_ylabel('Customer Tenure (months)', fontsize=10)

# Bar graph for Sum Insured
ax[2, 1].bar(clusters, sum_insured, color='lightyellow', edgecolor='black')
ax[2, 1].set_title('Sum Insured by Cluster', fontsize=12)
ax[2, 1].set_ylabel('Sum Insured', fontsize=10)

# Adjust layout for better spacing
plt.tight_layout(pad=3.0)  
plt.show()