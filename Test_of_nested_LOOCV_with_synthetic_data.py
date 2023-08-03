import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, GridSearchCV, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.feature_selection import SelectFromModel, SelectKBest, chi2, RFE, f_classif, mutual_info_classif, SelectFpr, SelectFdr, SelectFwe, SequentialFeatureSelector, RFECV, VarianceThreshold
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
#import sklearn_relief as sr
from boruta import BorutaPy

def CV_custom(X, y, cv_outer, cv_inner, pipeline, search_space):
    outer_results_accuracy = list()
    outer_y_pred = list()
    iteration = 1

    for train_indices, test_indices in cv_outer.split(X):
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]

        clf = GridSearchCV(pipeline, search_space, cv=cv_inner, verbose=0, n_jobs=3)
        result = clf.fit(X_train, y_train)
        best_model = result.best_estimator_

        y_pred = best_model.predict(X_test)
        outer_y_pred.append(y_pred)
        accuracy = accuracy_score(y_test,y_pred)
      
        outer_results_accuracy.append(accuracy)
        print('>iteration=%i, acc=%.3f, best=%.3f, cfg=%s' % (iteration ,accuracy, result.best_score_, result.best_params_))
        iteration = iteration + 1

    conf_mat = confusion_matrix(y, outer_y_pred)
    precision = precision_score(y, outer_y_pred, average = 'binary')
    recall = recall_score(y, outer_y_pred, average = 'binary')
    rocauc_macro = roc_auc_score(y, outer_y_pred, average = 'macro')
    rocauc_micro = roc_auc_score(y, outer_y_pred, average = 'micro')
    print('conf_mat: %s ' % (conf_mat))
    print('accuracy: %.3f ' % (np.mean(outer_results_accuracy)))
    print('precision: %.3f ' % (precision))
    print('recall: %.3f ' % (recall))
    print('rocauc_macro: %.3f ' % (rocauc_macro))
    print('rocauc_micro: %.3f ' % (rocauc_micro))
#=======================================================================================



seedValue = 48  #selector__k
''' BorutaPy(RandomForestClassifier(class_weight='balanced', max_depth=5), n_estimators='auto', verbose=2, random_state=seedValue)
    SelectKBest(chi2, k = 2),
    SelectKBest(f_classif, k = 2),
    RFE(estimator=LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seedValue), step = 0.1),
    SelectFromModel(estimator = RandomForestClassifier(n_estimators=250, criterion='gini',random_state=seedValue), threshold=-np.inf)]'''


'''[{'selector__k': [1,2,3,4,5,6,7,8]}],
                 [{'selector__k': [1,2,3,4,5,6,7,8]}],
                 [{'selector__n_features_to_select': [1,2,3,4,5,6,7,8]}],
                 [{'selector__max_features': [1,2,3,4,5,6,7,8]}]'''
#max_iter = 5000
#feature_selector, sel_name =  SelectKBest(chi2, k = 2), "SelectKBest_chi2_"
#feature_selector, sel_name =  SelectFromModel(estimator = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seedValue), threshold=-np.inf), "SFML2_"
#feature_selector, sel_name =  RFE(estimator=LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=seedValue), n_features_to_select = 2, step = 0.1), "RFEL1_"
#feature_selector, sel_name =  RFE(estimator=LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seedValue), n_features_to_select = 2, step = 0.1), "RFEL2_"
#feature_selector, sel_name =  SelectFromModel(estimator = RandomForestClassifier(n_estimators=250, criterion='gini',random_state=seedValue), threshold=-np.inf), "SFMRandForest_"
#feature_selector, sel_name =  SelectKBest(f_classif, k = 2), "SelectKBest_f_classif_"
#feature_selector, sel_name =  SelectFromModel(estimator = model, threshold=-np.inf, max_features = 2), "SelectFromModel_"
#feature_selector, sel_name =  VarianceThreshold(threshold = 0), "VarianceTreshold_"
#feature_selector, sel_name =  RFE(estimator=model, n_features_to_select = 2, step = 0.1), "RFE_"

#model, model_name = MLPClassifier(random_state=seedValue, hidden_layer_sizes=(100,), activation='relu', max_iter=5000), "MLP_" #no coef
#model, model_name = RandomForestClassifier(random_state=seedValue), "RandForest_" #yes coef
model, model_name = DecisionTreeClassifier(random_state=seedValue), "DecTree_" #yes coef
#model, model_name = LogisticRegression(max_iter=500), "LogRegression_" #yes coe
#model, model_name = LinearDiscriminantAnalysis(), "LinearDiAnalysis_" #?
#model, model_name = SVC(), "SVC_" #no coef

#scaler, sca_name = StandardScaler(), "Standard_"
scaler, sca_name = MinMaxScaler(), "MinMax_"


pipeline_production_model = Pipeline([ ('scaler', scaler),
                            ('classifier', model)])    

cv_outer = LeaveOneOut() 
#cv_inner , inner_cv_type= LeaveOneOut(), "LOO_" 
cv_inner, inner_cv_type = KFold(n_splits=10, shuffle=False), "CV10_" #KFold(n_splits=10, shuffle=False), "CV10_" 


feature_selector = SelectKBest(chi2, k = 2)
search_space = {'selector__k': [30,40,50,60]}


pipeline = Pipeline([
    ('scaler', scaler),
    ('selector', feature_selector),
    ('classifier', model)])

X, y = make_classification(n_samples=45,
                    n_features=1000,
                    n_informative=20,
                    flip_y= 0.9,
                    class_sep= 0.5,
                    n_classes=2)


#CV_custom(X, y, cv_outer, cv_inner, pipeline, search_space)


def excludeEmptyValues(y, columnName):
    codesOfRowsToDelete = y.loc[y[columnName].isnull(), 'kod']
    y.index_col = 0
    for index, row in codesOfRowsToDelete.iteritems():
        y = y[y['kod'] != row]
    return y

def GetDataFromFiles(data_source, decisionColumnName, binarize = True):
    geny_k = pd.DataFrame()
    cechy_k = pd.DataFrame()
    if(data_source == 'k_'):
        geny_k = pd.read_csv('./geny_krew.csv', sep=',', decimal=".", encoding="ANSI")
        cechy_k = pd.read_csv('./cechy_kliniczne_krew.csv', sep=',', decimal=".", encoding="ANSI")
    elif(data_source == 's_'):
        geny_k = pd.read_csv('./geny_szczotka.csv', sep=',', decimal=".", encoding="ANSI")
        cechy_k = pd.read_csv('./cechy_kliniczne_szczotka.csv', sep=',', decimal=".", encoding="ANSI")
    geny_k, cechy_k = excludeEmptyValues(geny_k, cechy_k, decisionColumnName)
    codes_of_objects = geny_k.iloc[:,:1]
    geny_k = geny_k.iloc[:,1:]
    cechy_k = cechy_k[decisionColumnName]
    if(binarize):
        cechy_k = (cechy_k.astype(float) >= cechy_k.median()).astype(int) #binaryzacja
    print(cechy_k.value_counts())
    print(cechy_k.shape)
    print(geny_k.shape)
    geny_k_column_names = list(geny_k.columns.values)
    return geny_k, cechy_k, geny_k_column_names, codes_of_objects


def GetObjectsOnChart(data_source, decisionColumnName):
    cechy_k_original = pd.DataFrame()
    #cechy_k_binarized = pd.DataFrame()
    if(data_source == 'k_'):
        cechy_k = pd.read_csv('./cechy_krew.csv', sep=',', decimal=".", encoding="ANSI")
    elif(data_source == 's_'):
        cechy_k = pd.read_csv('./cechy_szczotka.csv', sep=',', decimal=".", encoding="ANSI")
    print(cechy_k.shape)
    cechy_k = excludeEmptyValues(cechy_k, decisionColumnName)
    cechy_k = cechy_k[['kod', decisionColumnName]]
    print(cechy_k.sort_values(by=[decisionColumnName]))

GetObjectsOnChart('k_', 'Kolagen_I_sila')