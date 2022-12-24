import pandas as pd
import modelselect as ms
from sklearn.model_selection import train_test_split

# read data
url = 'https://raw.githubusercontent.com/shoukewei/data/main/data-pydm/gdp_china_encoded.csv'
df = pd.read_csv(url,index_col=False)

# Define independent variables (X) and dependent variable (y)
X = df.drop(['gdp'],axis=1)
y = df['gdp']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.30, random_state=1)

# drop pop variable
X_drop = ['pop']

# display the descriptive statistic mesures in Pandas DataFrame
modelres = ms.LRSelector(X_train,y_train,X_drop)

# display the results
res = modelres[0]
vif = modelres[1]
print(res.summary())
print(vif)