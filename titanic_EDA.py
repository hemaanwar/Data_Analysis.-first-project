# 1 imporing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
sns.set()
#  exploring data

df = pd.read_csv('train.csv')
print(df.head())
# unuseful feture --> passID / Name / Ticket
# categorical feture --> sex / pcclass / sibsp / parch / embarked
# numerical feture --> Age / fare
# target --> survived
print(df.info())
print(df.describe())
print(df.describe(include='O'))
# check dublicated data
print(df.duplicated().sum())
# check missing values 
print(df.isnull().sum())
print(round(df.isna().mean()*100,2))

df.dropna(subset=['Embarked'],inplace=True)
print(df.isnull().sum()) 

sns.histplot(x='Age',data=df)
plt.show()
print(df.groupby(['Pclass','Sex']).Age.median())
df['Age']=df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

# imputer = KNNImputer(n_neighbors=2)
# df[["Age"]]= imputer.fit_transform(df[['Age']])
# print(df.isnull().sum())

df.drop([ 'Cabin','Ticket','PassengerId'], axis=1 ,inplace=True)
print(df.isnull().sum()) 
print(df.duplicated().sum())

df.columns = df.columns.str.lower()
print(df.sample())

# uni variant analysis

print(df.info())
print(df.survived.value_counts(normalize=True))
print(df.survived.value_counts())

def explore_categorical (df,col):
    print(f'### {col} ###')
    print(df[col].value_counts(normalize=True))
    plt.title(f'count plot of {col}')
    sns.countplot(x=col,data=df)
    plt.show()

for col in ['survived','pclass','sex','sibsp','parch']:
    explore_categorical(df,col)


def explore_continuous (df,col):
    print(f'### {col} ###')
    print(df[col].describe())
    plt.title(f'hist plot of {col}')
    sns.histplot(x=col,data=df)
    plt.show()

for col in ['age','fare']:
    explore_continuous(df,col)

# outlier detection

sns.boxplot(x='age',data=df)
plt.show()
sns.boxplot(x='fare',data=df)
plt.show()

df =df[df.fare<300]

sns.boxplot(x='fare',data=df)
plt.show()
print(df.fare.describe())

# Bi variant analysis

print(df.sample())

def servival_rate (df,col):
    print(f'### {col} ###')
    print(df.groupby(col).survived.mean())
    plt.title(f'barplot of {col}')
    sns.barplot(x=col , y='survived', data=df, ci=None)
    plt.axhline(df.survived.mean(), color='black',linestyle='--')
    plt.show()
    
for col in ['pclass','sex','sibsp','parch','embarked']:
    servival_rate(df,col)

sns.histplot(x="age",data=df,hue='survived',multiple='stack')
plt.show()
sns.histplot(x="fare",data=df,hue='survived',multiple='stack')
plt.show()

df_survived = df[df.survived==1]
df_died = df[df.survived==0]
fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.histplot(x='age', data=df_survived, ax=ax[0], kde=True, color='green')
sns.histplot(x='age', data=df_died, ax=ax[1], kde=True, color='red')
ax[0].set_title('Survived')
ax[1].set_title('Died')
plt.show()

fig, ax = plt.subplots(1, 2, figsize=(12,4))
sns.histplot(x='fare', data=df_survived, ax=ax[0], kde=True, color='green')
sns.histplot(x='fare', data=df_died, ax=ax[1], kde=True, color='red')
ax[0].set_title('Survived')
ax[1].set_title('Died')
plt.show()

# split continouse data to group

print(df.describe()[['age','fare']])

df['age_group']=pd.cut( df.age, bins=[ 0, 20, 30, 39, 82], labels=[ 'child', 'young', 'adult', 'senior'])
df['fare_group']=pd.cut( df.fare, bins=[ -0.99, 8, 15, 35, 265], labels=[ 'low', 'medium', 'high', 'very high'])

for col in ['age_group','fare_group']:
    servival_rate(df,col)

# multi variant

print(df.select_dtypes(include='number').corr()['survived'])

sns.heatmap(data=df.select_dtypes(include='number').corr(),annot=True)
plt.show()

sns.barplot(x='pclass',y='survived',hue='sex',data=df,ci=None)
plt.show()

sns.barplot(x='embarked',y='survived',hue='sex',data=df,ci=None)
plt.show()

# conclusion

fig , ax = plt.subplots(2,4,figsize=(15,8))
for i, col in enumerate(['pclass','sibsp','parch','age_group','fare_group','embarked','sex']):
    sns.barplot(x=col, y='survived', data=df, ci=None, ax=ax[i//4, i%4])
    ax[i//4, i%4].axhline(df.survived.mean(), color='black',linestyle='--')
plt.tight_layout()
plt.show()

femail_df =df[df.sex == 'female']
mail_df =df[df.sex == 'male']

print(df.groupby(['pclass','sex']).survived.mean())

sns.barplot(x='pclass',y='survived',data=mail_df, ci=None)
plt.axhline(mail_df.survived.mean() , color='black',linestyle='--')
plt.show()
sns.barplot(x='pclass',y='survived',data=femail_df, ci=None)
plt.axhline(femail_df.survived.mean() , color='black',linestyle='--')
plt.show()





