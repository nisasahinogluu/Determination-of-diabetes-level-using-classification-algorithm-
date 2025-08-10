# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

# 1. Load the Dataset
file_path = '/Users/nisasahinoglu/Desktop/Diyabet/Dataset of Diabetes .csv'
data = pd.read_csv(file_path, encoding='latin1')

print("1. First 5 rows of the original dataset:")
print(data.head())

# 2. Data Cleaning and Preprocessing


# 2.1 - Check for missing values
print("\n2.1 - Missing values in each column:")
print(data.isnull().sum())

# 2.2 Unique values in each column
print("\nUnique values in class, gender columns:")
print("Class: ",data['CLASS'].unique())
print("Gender: ",data['Gender'].unique())


# 2.3 Change values of CLASS ('N '-> 'N', 'Y '-> 'Y') and Gender columns ('f' -> 'F')

data['CLASS'] = data['CLASS'].replace({'N ': 'N', 'Y ': 'Y'})
data['Gender'] = data['Gender'].replace({'f': 'F'})

print("\n2.2 - After replacing values in CLASS and Gender columns:")
print("Class: ",data['CLASS'].unique())
print("Gender: ",data['Gender'].unique())

# 2.4 convert CLASS labels to numeric

class_mapping = {'N': 0, 'Y': 1, 'P': 2}
data['CLASS'] = data['CLASS'].map(class_mapping)
print("\n2.3 - After converting CLASS labels to numeric:")
print(data.head())

# 2.5 Convert Gender to numeric
data['Gender'] = data['Gender'].map({'F': 0, 'M': 1})

print("\n2.6 - After converting Gender to numeric:")
print(data.head())

# 3. Feature Selection using Mutual Information
X = data.drop(columns=['CLASS'])
y = data['CLASS']

mi_scores = mutual_info_classif(X, y, discrete_features='auto', random_state=42)
mi_scores_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
ax = sns.barplot(x=mi_scores_series.values, y=mi_scores_series.index, palette='viridis')
if ax.legend_: ax.legend_.remove()
plt.title('Mutual Information Scores of All Features')
plt.xlabel('Mutual Information Score')
plt.ylabel('Features')
plt.tight_layout()
plt.savefig('mutual_information_scores_lmch.png', dpi=300)
plt.show()

# 5. Print and select the top 5 features with highest MI scores
selected_features = mi_scores_series.head(5).index.tolist()
print("\n5. Top 5 features with highest Mutual Information scores:")
for feature in selected_features:
    print(f"{feature}: {mi_scores_series[feature]:.4f}")

# 6. Create a new dataset with selected features and save it
final_data = data[selected_features + ['CLASS']]

print("\n6.1 - First 5 rows of the new dataset with selected features and CLASS:")
print(final_data.head())

final_data.to_csv('lmch_cleaned_selected.csv', index=False)

print("\n6.2 - The dataset has been successfully cleaned and saved with selected features.")