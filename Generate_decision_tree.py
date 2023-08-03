from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
import pandas as pd
import numpy as np
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

def GetDataFromFiles(data_source, decisionColumnName, id_to_remove, binarize = True):
    geny_k = pd.DataFrame()
    cechy_k = pd.DataFrame()
    if(data_source == 'k_'):
        geny_k = pd.read_csv('./geny_krew.csv', sep=',', decimal=".", encoding="ANSI")
        cechy_k = pd.read_csv('./cechy_krew.csv', sep=',', decimal=".", encoding="ANSI")
    elif(data_source == 's_'):
        geny_k = pd.read_csv('./geny_szczotka.csv', sep=',', decimal=".", encoding="ANSI")
        cechy_k = pd.read_csv('./cechy_szczotka.csv', sep=',', decimal=".", encoding="ANSI")
    geny_k, cechy_k = excludeEmptyValues(geny_k, cechy_k, decisionColumnName)
    
    x_target = geny_k[geny_k.kod == id_to_remove]
    y_target = cechy_k[cechy_k.kod == id_to_remove]
    median =  cechy_k[decisionColumnName].median()

    geny_k.drop(geny_k[geny_k.kod == id_to_remove].index, inplace=True)
    cechy_k.drop(cechy_k[cechy_k.kod == id_to_remove].index, inplace=True)

    geny_k = geny_k.iloc[:,1:]
    cechy_k = cechy_k[decisionColumnName]
    y_target = y_target[decisionColumnName]
    if(binarize):
        cechy_k = (cechy_k.astype(float) >= median).astype(int) #binaryzacja
        y_target = (y_target.astype(float) >= median).astype(int) #binaryzacja
    else:
        cechy_k = cechy_k.astype(int)
    print(cechy_k.value_counts())
    print(cechy_k.shape)
    print(geny_k.shape)
    return geny_k, cechy_k, x_target, y_target


def excludeEmptyValues(X, y, columnName):
    codesOfRowsToDelete = y.loc[y[columnName].isnull(), 'kod']
    y.index_col = 0
    X.index_col = 0
    for index, row in codesOfRowsToDelete.iteritems():
        X = X[X['kod'] != row]
        y = y[y['kod'] != row]
    return X, y

decisionColumnName, decColumnName = "Kolagen_I_proc_pow", "Kolagen_I_pow"
#decisionColumnName, decColumnName = "Kolagen_I_sila","Kolagen_I_sila"
#decisionColumnName, decColumnName = "wall_area_ratio_RB1", "RB1"
#decisionColumnName, decColumnName = "wall_area_ratio_RB10", "RB10"
#decisionColumnName, decColumnName = "wall_thichness_airway_diameter_ratio_RB1", "Wall_thic_RB1"
#decisionColumnName, decColumnName = "srednia_harmoniczna_liniowa", "Srednia_harmoniczna"

clf = DecisionTreeClassifier(random_state=48)
scaler = MinMaxScaler()

id_to_remove= "AR45_S"
selected_features = ['spot5713']

X, y, x_target, y_target = GetDataFromFiles("s_", decisionColumnName, id_to_remove, False)

X = X[selected_features]
x_target = x_target[selected_features]

X = scaler.fit_transform(X)
x_target =scaler.transform(x_target)

#sampler = SMOTE(random_state=seedValue, k_neighbors=3)

model = clf.fit(X, y)

both = X.copy()
y_copy = y.to_numpy()
both = np.append(both, y_copy[:,None], axis=1)
print(both)
print(y)
print("target Object:")
print(x_target)
print(y_target)

fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(clf, 
                   feature_names= selected_features,  
                   class_names= ['0','1'],
                   filled=True)
fig.savefig("decistion_tree.png")