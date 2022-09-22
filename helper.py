import pandas as pd
from Models import ANN

a = ANN((2,3), "sigmoid")

y = pd.Series([1,2,2,1])
X = pd.DataFrame({'a': [0.35,1.7,-1,0.2],'b':[1,5,8,1.6],'c':[1.2,5,-1.4,-3.6],'d':[1,5,8,1.6],'e':[1.2,5,-1.4,-3.6]})

a.fit(X,y)