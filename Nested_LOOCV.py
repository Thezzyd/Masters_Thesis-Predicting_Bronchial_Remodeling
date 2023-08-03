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
from imblearn.pipeline import Pipeline
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score, balanced_accuracy_score, make_scorer, f1_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
#import sklearn_relief as sr
from boruta import BorutaPy

def GetDataFromFiles(data_source, decisionColumnName, binarize = True):
    geny_k = pd.DataFrame()
    cechy_k = pd.DataFrame()
    if(data_source == 'k_'):
        geny_k = pd.read_csv('./Datasets/geny_krew.csv', sep=',', decimal=".", encoding="ANSI")
        cechy_k = pd.read_csv('./Datasets/cechy_kliniczne_krew.csv', sep=',', decimal=".", encoding="ANSI")
    elif(data_source == 's_'):
        geny_k = pd.read_csv('./Datasets/geny_szczotka.csv', sep=',', decimal=".", encoding="ANSI")
        cechy_k = pd.read_csv('./Datasets/cechy_kliniczne_szczotka.csv', sep=',', decimal=".", encoding="ANSI")
    geny_k, cechy_k = excludeEmptyValues(geny_k, cechy_k, decisionColumnName)
    codes_of_objects = geny_k.iloc[:,:1]
    geny_k = geny_k.iloc[:,1:]
    cechy_k = cechy_k[decisionColumnName]
    if(binarize):
        cechy_k = (cechy_k.astype(float) >= cechy_k.median()).astype(int) #binaryzacja
    else:
        cechy_k = cechy_k.astype(int)
    print(cechy_k.value_counts())
    print(cechy_k.shape)
    print(geny_k.shape)
    geny_k_column_names = list(geny_k.columns.values)
    return geny_k, cechy_k, geny_k_column_names, codes_of_objects


def SaveLineOfText(file_name, text_to_save):
    with open("./Data_Exploration_Results/"+str(file_name)+'.txt', 'a') as the_file:
        the_file.write(str(text_to_save)+'\n')

def excludeEmptyValues(X, y, columnName):
    codesOfRowsToDelete = y.loc[y[columnName].isnull(), 'kod']
    y.index_col = 0
    X.index_col = 0
    for index, row in codesOfRowsToDelete.iteritems():
        X = X[X['kod'] != row]
        y = y[y['kod'] != row]
    return X, y

def VoteOnFeaturesToProductionModel(count_of_selected_features, all_features_summarized_int):
    count_of_selected_features_sorted = dict(sorted(count_of_selected_features.items(), key=lambda x:x[1], reverse=True))
    print(count_of_selected_features_sorted)
    SaveLineOfText(filename_to_save, "\n\n Podliczone_cechy: "+str(count_of_selected_features_sorted))
    selected_features_list = list(count_of_selected_features_sorted.keys())
    voted_features_int =  all_features_summarized_int / len(cechy_k)
    print('Wybrane cechy: '+str(selected_features_list[:round(voted_features_int)]))
    X_selected_features = geny_k[selected_features_list[:round(voted_features_int)]]
    SaveLineOfText(filename_to_save, "\n Wybrane_cechy: "+str(selected_features_list[:round(voted_features_int)]))
    return X_selected_features

def RunCvOnProductionModel(X_selected_features, y, pipeline, cv):
    y_pred = cross_val_predict(pipeline, X_selected_features, y, cv=cv)
    conf_matrix_production_model = confusion_matrix(y, y_pred)
    accuracy_production_model = accuracy_score(y, y_pred)
    precision_production_model = precision_score(y, y_pred, average = 'binary')
    recall_production_model = recall_score(y, y_pred, average = 'binary')
    #roc_auc_macro_production_model = roc_auc_score(y, y_pred, average = 'macro')
    #roc_auc_micro_production_model = roc_auc_score(y, y_pred, average = 'micro')
    balanced_accuracy_score_production_model = balanced_accuracy_score(y, y_pred)
    print('Production model:')
    print('conf_matrix: '+str(conf_matrix_production_model))
    print('accuracy: '+str(accuracy_production_model))
    print('precision: '+str(precision_production_model))
    print('recall: '+str(recall_production_model))
    #print('roc_auc_macro: '+str(roc_auc_macro_production_model))
    #print('roc_auc_micro: '+str(roc_auc_micro_production_model))
    print('balanced accuracy: '+str(balanced_accuracy_score_production_model))

    
    SaveLineOfText(filename_to_save, "\n Produkcyny_model_uzyskal: \n"+"conf_matrix:"+str(conf_matrix_production_model)+"\n accuracy:"+str(np.mean(accuracy_production_model))+"\n precision:"+str(precision_production_model)+"\n recall:"+str(recall_production_model)+"\n balanced_acc:"+str(balanced_accuracy_score_production_model))
    SaveLineOfText(filename_to_save, "\n\n=============================================")

def CV_custom(X, y, cv_outer, cv_inner, pipeline, search_space,filename_to_save, X_column_names = None, codes_of_objects = None):
    outer_results_accuracy = list()
    outer_y_pred = list()
    all_selected_features = []
    codes_of_objects_misclassified = []
    codes_of_objects_classified_correctly = []
    iteration = 1

    SaveLineOfText(filename_to_save, "=============================================")
    SaveLineOfText(filename_to_save, "search_space: "+str(search_space))

    for train_indices, test_indices in cv_outer.split(X):
        '''X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]'''
        X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
        y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
        scorer = make_scorer(f1_score, average = 'weighted')
        clf = GridSearchCV(pipeline, search_space, cv=cv_inner, verbose=0, n_jobs=-1, scoring=scorer, error_score='raise')
        result = clf.fit(X_train, y_train)
        best_model = result.best_estimator_

        feature_names = best_model.named_steps.selector.get_feature_names_out(X_column_names)
        all_selected_features.extend(feature_names)

        y_pred = best_model.predict(X_test)
        outer_y_pred.append(y_pred)
        accuracy = accuracy_score(y_test,y_pred)
        if(int(accuracy == 0)):
            codes_of_objects_misclassified.append(str(codes_of_objects.iloc[iteration-1].values.ravel()))
        else:
            codes_of_objects_classified_correctly.append(str(codes_of_objects.iloc[iteration-1].values.ravel()))
        
        outer_results_accuracy.append(accuracy)
        print('>iteration=%i, acc=%.3f, best=%.3f, cfg=%s, test_object_id=%s, feature_names=%s' % (iteration ,accuracy, result.best_score_, result.best_params_, codes_of_objects.iloc[iteration-1].values.ravel(), feature_names))
        SaveLineOfText(filename_to_save, ">iteration="+str(iteration)+", acc="+str(accuracy)+", best="+str(result.best_score_)+", cfg="+str(result.best_params_)+", test_object_id="+str(codes_of_objects.iloc[iteration-1].values.ravel())+", feature_names="+str(feature_names))
        iteration = iteration + 1

    count_of_selected_features = dict((x,all_selected_features.count(x)) for x in set(all_selected_features))
    conf_mat = confusion_matrix(y, outer_y_pred)
    precision = precision_score(y, outer_y_pred, average = 'binary')
    recall = recall_score(y, outer_y_pred, average = 'binary')
    #rocauc_macro = roc_auc_score(y, outer_y_pred, average = 'macro')
    #rocauc_micro = roc_auc_score(y, outer_y_pred, average = 'micro')
    balanced_acc = balanced_accuracy_score(y, outer_y_pred)

    print('conf_mat: %s ' % (conf_mat))
    print('accuracy: %.3f ' % (np.mean(outer_results_accuracy)))
    print('precision: %.3f ' % (precision))
    print('recall: %.3f ' % (recall))
    #print('rocauc_macro: %.3f ' % (rocauc_macro))
    #print('rocauc_micro: %.3f ' % (rocauc_micro))
    print('balanced_accuracy: %.3f ' % (balanced_acc))

    SaveLineOfText(filename_to_save, "Results: \n"+"conf_matrix:"+str(conf_mat)+"\n accuracy:"+str(np.mean(outer_results_accuracy))+"\n precision:"+str(precision)+"\n recall:"+str(recall)+"\n balanced_acc:"+str(balanced_acc))
    SaveLineOfText(filename_to_save, "Misclassified objects are:"+str(codes_of_objects_misclassified)+"\n Correctly classified objects are:"+str(codes_of_objects_classified_correctly))
    return count_of_selected_features, len(all_selected_features)
#=======================================================================================


# [{'selector__k': [13,14,15,16,17]}]
# "Kolagen_I_proc_pow", "wall_area_ratio_RB1", "wall_area_ratio_RB10", "wall_thichness_airway_diameter_ratio_RB1", "srednia_harmoniczna_liniowa"
# "Kolagen_pow_", "RB1_", "RB10_", "Wall_thich_RB1_", "sr_harmoniczna_"
namesOfColumns = ["wall_area_ratio_RB1"]
namesOfColumnsShort = ["wall_area_ratio_RB1"]
#namesOfColumns = ["wall_thichness_airway_diameter_ratio_RB1", "srednia_harmoniczna_liniowa"]
#namesOfColumnsShort = ["Wall_thich_RB1_", "Sr_harmoniczna_"]


search_spaces = [{'selector__k': [1,2,3,4]},
                {'selector__k': [1,2,3,4]},
                {'selector__max_features': [1,2,3,4]},
                {'selector__n_features_to_select': [1,2,3,4]},
                {'selector__n_features_to_select': [1,2,3,4]}]
 #, [{'selector__n_features_to_select': [2,3,4,5,6]}], [{'selector__max_features': [2,3,4,5,6]}]


seedValue = 48  #selector__k
data_source = "s_"
selectors = [SelectKBest(chi2, k = 2),
            SelectKBest(f_classif, k = 2),
            SelectFromModel(estimator = RandomForestClassifier(n_estimators=100, criterion='gini',random_state=seedValue), threshold=-np.inf),
            RFE(estimator=LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seedValue), step = 0.1),
            RFE(estimator=LogisticRegression(C=1.0, penalty='l1', solver='liblinear', random_state=seedValue), step = 0.1)]

''' BorutaPy(RandomForestClassifier(class_weight='balanced', max_depth=5), n_estimators='auto', verbose=2, random_state=seedValue)
    SelectKBest(chi2, k = 2),
    SelectKBest(f_classif, k = 2),
    RFE(estimator=LogisticRegression(C=1.0, penalty='l2', solver='liblinear', random_state=seedValue), step = 0.1),
    SelectFromModel(estimator = RandomForestClassifier(n_estimators=250, criterion='gini',random_state=seedValue), threshold=-np.inf)]'''

selectorNames = ["SelectKBest_chi2_",
    "SelectKBest_f_classif",
    "SFMRandForest_",
    "RFEL2_",
    "RFEL1_"]
'''"
    "Boruta_"
    "SelectKBest_chi2_",
    "SelectKBest_f_classif",
    "RFEL2_",
    "SFMRandForest_""'''


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
#model, model_name = DecisionTreeClassifier(random_state=seedValue), "DecTree_" #yes coef
#model, model_name = LogisticRegression(max_iter=500), "LogRegression_" #yes coe
#model, model_name = LinearDiscriminantAnalysis(), "LinearDiAnalysis_" #?
#model, model_name = SVC(), "SVC_" #no coef
model, model_name = GradientBoostingClassifier(), "GradientBoost_" #no coef
#scaler, sca_name = StandardScaler(), "Standard_"
scaler, sca_name = MinMaxScaler(), "MinMax_"
#sampler = SMOTE(random_state=seedValue, k_neighbors=3)

pipeline_production_model = Pipeline([('scaler', scaler),                         
                            ('classifier', model)])    

cv_outer = LeaveOneOut() 
#cv_inner , inner_cv_type= LeaveOneOut(), "LOO_" 
cv_inner, inner_cv_type =LeaveOneOut(), "LOO_" #KFold(n_splits=10, shuffle=False), "CV10_" 

for i in range(len(namesOfColumns)):
    for j in range(len(selectors)):
        decisionColumnName, col_name = namesOfColumns[i], namesOfColumnsShort[i]
        feature_selector, sel_name, search_space = selectors[j], selectorNames[j], search_spaces[j]

        pipeline = Pipeline([
            ('scaler', scaler),
            ('selector', feature_selector),
            ('classifier', model)])
        
        filename_to_save = str(data_source+col_name+model_name+sel_name+sca_name+inner_cv_type)
        geny_k, cechy_k, geny_k_column_names, codes_of_objects = GetDataFromFiles(data_source, decisionColumnName, binarize=True)

        count_of_selected_features, all_features_summarized_int = CV_custom(geny_k, cechy_k, cv_outer, cv_inner, pipeline, search_space, filename_to_save, X_column_names = geny_k_column_names, codes_of_objects = codes_of_objects)

        X_selected_features = VoteOnFeaturesToProductionModel(count_of_selected_features, all_features_summarized_int)
        RunCvOnProductionModel(X_selected_features, cechy_k, pipeline_production_model, LeaveOneOut())
