import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


file = r'C:\Users\admin\Desktop\Stage4eme\Code\CompositionPortefeuille.xls'

df = pd.read_excel(file)

# Describe data 
print(df.head())  
print(df.info())
print(df.describe())

# Renaming columns
df.rename(columns={
    "Date de naissance de l'assuré": "DateNaissance",
    "Titre de civilité de l'assuré": "Sexe",
    "Ville de l'assuré": "Ville",
    "Encaissements": "Encaissements",
    "Date d'effet": "DateEffet",
    # Add other column renaming here if needed
}, inplace=True)

# Convert 'Sexe' to 'Male' if 'Monsieur', 'Female' if 'Madame'
df['Sexe'] = df['Sexe'].apply(lambda x: 'Male' if x == 'Monsieur' else 'Female')

# Check for missing values
print(df.isnull().sum())

# Dropping NA Values
df = df.dropna()

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Display the first few rows of the encoded dataset
print(df.head())

# Select relevant columns for plotting
relevant_columns = ['DateNaissance', 'Sexe', 'Encaissements', 'DateEffet', 'Ville']
df_relevant = df[relevant_columns]

# Plot distribution of numerical columns
numerical_cols = df_relevant.select_dtypes(include=['int64', 'float64']).columns


plt.figure(figsize=(15, 10))
for i, col in enumerate(numerical_cols, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df_relevant[col], kde=True)
    plt.title(f'Distribution of {col}')
plt.tight_layout()
plt.show()

# Plot correlation matrix for relevant columns
plt.figure(figsize=(8, 6))
corr_matrix = df_relevant.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Relevant Columns')
plt.show()

# Pairplot for numerical columns to see relationships
sns.pairplot(df_relevant[numerical_cols])
plt.show()

# Plotting DateNaissance, Sexe, and Ville with labels
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
df['DateNaissance'].dt.year.value_counts().sort_index().plot(kind='bar')
plt.title('Distribution of DateNaissance')
plt.xlabel('Year')
plt.ylabel('Count')

plt.subplot(3, 1, 2)
sns.countplot(x='Sexe', data=df)
plt.title('Distribution of Sexe')
plt.xlabel('Sexe')
plt.ylabel('Count')

plt.subplot(3, 1, 3)
df['Ville'].value_counts().plot(kind='bar')
plt.title('Distribution of Ville')
plt.xlabel('Ville')
plt.ylabel('Count')

plt.tight_layout()
plt.show()