import pandas as pd 
import numpy as np
import math
import sys
col_names=['age','workclass','fnlwgt','education','education_num',
'marital_status','occupation','relationship','race','sex',
'capital_gain','capital_loss','hours_per_week','native_country','high_income']
income=pd.read_table('./data/income.data',sep=',',names=col_names)


#sys.setrecursionlimit(20)


#处理数据
columns=['workclass','education','marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country','high_income']
for name  in columns:
	col=pd.Categorical.from_array(income[name])
	income[name]=col.codes

#Splitting data
private_incomes=income[income['workclass']==4]
public_incomes=income[income['workclass']!=4]

#Calculating_entropy
def calc_entropy(column):
	counts=np.bincount(column)
	probabilities=counts/len(column)
	entropy=0
	for prob in probabilities:
		if prob>0:
			entropy+=prob*math.log(prob,2)
	return -entropy

#Calculating information_gain
def calc_information_gain(data,split_name,target_name):
	original_entropy=calc_entropy(data[target_name])
	column=data[split_name]
	median=column.median()
	
	left_split=data[column<=median]
	right_split=data[column>median]
	
	to_subtract=0
	for subset in [left_split,right_split]:
		prob=(subset.shape[0]/data.shape[0])
		to_subtract+=prob*calc_entropy(subset[target_name])
	return original_entropy - to_subtract	

#Finding best split column
def find_best_column(data,columns,target_column):

	information_gains=[]
	for col in columns:
		information_gains.append(calc_information_gain(data,col,target_column))
	highest_gain=columns[information_gains.index(max(information_gains))]
	return highest_gain
	



#Constructing DecisionTree-using id3 algorithm and storing it .

def id3(data,columns,target,tree):
	unique_targets=pd.unique(data[target])
	
	nodes.append(len(nodes)+1)
	tree['number']=nodes[-1]
	if len(unique_targets)==1 :
		tree['label']=unique_targets[0]
		return tree  
	
	best_column=find_best_column(data,columns,target)
	column_median=data[best_column].median()
	
	tree['column']=best_column
	tree['median']=column_median
	
	left_split=data[data[best_column] <= column_median]
	right_split=data[data[best_column] > column_median]

	split_dict=[["left",left_split],["right",right_split]]
	for name,split in split_dict:
		tree[name]={}
		id3(split,columns,target,tree[name])



#Printing a more attractive tree
def print_with_depth(string,depth):
	prefix="   "*depth
	print("{0}{1}".format(prefix,string))
def print_node(tree,depth):
	if 'label' in tree:
		print_with_depth("Leaf:Label {0}".format(tree['label']),depth)
		return
	print_with_depth("{0}>{1}".format(tree['column'],tree['median']),depth)
	branches=[tree['left'],tree['right']]
	for branch in branches:
		print_node(branch,depth+1)

#Making predictions
def predict(tree,row):
	if 'label' in tree:
		return tree['label']
	column=tree['column']
	median=tree['median']
	if row[column]<=median:
		return predict(tree['left'],row)
	else:
		return predict(tree['right'],row)
def batch_predict(tree,df):
	predictions=df.apply(lambda x:predict(tree,x),axis=1)
	return predictions

columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

tree={} 
nodes=[] #保存节点编号

data = pd.DataFrame([
    [0,20,0],
    [0,60,2],
    [0,40,1],
    [1,25,1],
    [1,35,2],
    [1,55,1]
    ])
data.columns = ["high_income", "age", "marital_status"]

'''new_data = pd.DataFrame([
    [40,0],
    [20,2],
    [80,1],
    [15,1],
    [27,2],
    [38,1]
    ])
new_data.columns=["age", "marital_status"]'''
train=income[:100]  #预测全部时发生栈溢出现象，所以只预测部分数据
test=income[100:110]

#id3(income,['age','marital_status'],'high_income',tree)
id3(train,columns,'high_income',tree)

print("======>>>Decision Tree:")
print(print_node(tree,0))

actual_prediction=pd.DataFrame({'actual':test['high_income'] ,'pred':batch_predict(tree,test)})
actual_prediction.index=range(10)

print("=====>>>>>预测：")
print(actual_prediction)





	
	
	
	
	
