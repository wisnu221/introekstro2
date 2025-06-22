# Kode ini untuk dijalankan di sel Google Colab

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import pickle

# 1. Muat dataset dari file yang sudah di-upload
df = pd.read_csv('personality_dataset.csv')

# 2. Persiapan Data (Preprocessing)
numerical_cols = df.select_dtypes(include=np.number).columns
categorical_cols = df.select_dtypes(include='object').columns.drop('Personality')

num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='most_frequent')

df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

df['Stage_fear'] = df['Stage_fear'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Drained_after_socializing'] = df['Drained_after_socializing'].apply(lambda x: 1 if x == 'Yes' else 0)

le = LabelEncoder()
df['Personality'] = le.fit_transform(df['Personality'])

X = df.drop('Personality', axis=1)
y = df['Personality']

# 3. Latih Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# 4. Simpan Model dan Objek Preprocessing ke file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('num_imputer.pkl', 'wb') as f:
    pickle.dump(num_imputer, f)
with open('cat_imputer.pkl', 'wb') as f:
    pickle.dump(cat_imputer, f)
with open('le.pkl', 'wb') as f:
    pickle.dump(le, f)
with open('columns.pkl', 'wb') as f:
    pickle.dump(X.columns, f)

print("Model dan semua file .pkl berhasil dibuat di environment Colab!")
print("Silakan cek panel 'Files' di sebelah kiri.")
