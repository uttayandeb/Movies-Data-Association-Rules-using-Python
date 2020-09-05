###############################################################################
###################        Association Rules       ############################



#importing packages and loading the data
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


movies=pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Assignment(Association1)\\my_movies.csv",sep=",")
movies = movies.iloc[:,[0,1,2,3,4]]
# Considering only the transactions, i.e the textual transactions.

#Imputation to convert the nan values to 0's
movies.iloc[:,2:5] = movies.iloc[:,2:5].apply(lambda x:x.fillna(0))

X = pd.get_dummies(movies[['V1','V2','V3','V4','V5']])

######### Creating model with dummies of NAN values #################3

x_dummies = X.iloc[:,[9,14,16]]

# Running Apriori algorithm
# With support=0,005 and max_len=2
frequent_items = apriori(x_dummies,min_support = 0.005, max_len =2 , use_colnames = True )
frequent_items.sort_values('support', ascending = False, inplace = True)

# Building rules
# with min_threshold =1
rules_dummies = association_rules(frequent_items, metric = 'lift', min_threshold = 1)
rules_dummies.sort_values('lift',ascending =False,inplace =True)

########## To eliminate redudancy in rules ########

def to_list(i):
    return(sorted(i))
    
rules_add = rules_dummies.antecedents.apply(to_list) + rules_dummies.consequents.apply(to_list)

rules_add = rules_add.apply(sorted)

rules_set = list(rules_add)

unique_rules = [list(m) for m in set(tuple(i) for i in rules_set)]
index_rules = []
for i in unique_rules:
    index_rules.append(rules_set.index(i))
    
# rules without redudancy
rules_without_redud = rules_dummies.iloc[index_rules,:]

# sorting
rules_without_redud.sort_values('lift', ascending = False, inplace =True)

# Support and confidence
Support = rules_without_redud['support']
confidence = rules_without_redud['confidence']

import matplotlib.pyplot as plt

plt.scatter(Support,confidence);plt.xlabel("Support");plt.ylabel("Confidence")


# Model with other than Zero values
x_without_dum = X.iloc[:,[0,1,2,3,4,5,6,7,8,10,11,12,13,15,17]]

#Applying Apriori with support = 0.005 and max length =2
frequents_item1 = apriori(x_without_dum, min_support=0.005, max_len = 2, use_colnames=True)
frequents_item1.sort_values('support',ascending = False, inplace = True)

# Building rules with minimum threshold = 1
rules_without = association_rules(frequents_item1, metric='lift', min_threshold =1)
rules_without.sort_values('lift',ascending = False, inplace =True)

# Eliminate the reducdancy####
def to_list_out(i):
    return(sorted(i))
    
rules_out_add = rules_without.antecedents.apply(to_list_out)+rules_without.consequents.apply(to_list_out)

rules_out_add = rules_out_add.apply(sorted)
rules_set_out = list(rules_out_add)

# unique values
unique_values_out = [list(n) for n in set(tuple(i) for i in rules_set_out)]
index_rules_out=[]
for i in unique_values_out:
    index_rules_out.append(rules_set_out.index(i))

# rules without redundancy
rules_without_out = rules_without.iloc[index_rules_out,:]

# Sorting
rules_without_out.sort_values('lift', ascending= False, inplace =True)

Support_out = rules_without_out["support"]
Confidence_out = rules_without_out["confidence"]
lift = rules_without_out["lift"]

#### Plotting 3D plot for support, confidence and lift
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection = "3d")
ax.scatter(Support_out, Confidence_out, lift)
ax.set_xlabel("Support_out")
ax.set_ylabel("Confidence_out")
ax.set_zlabel("lift")


#### scatter plot for rules for support, confidence and lift
import matplotlib.pyplot as plt
import scipy as sp

plt.scatter(Support_out, Confidence_out,c= lift,cmap='gray')
plt.colorbar()
plt.xlabel("Support")
plt.ylabel("Confidence")


# Creating the model with the dummy variables given in the data set itself#####
# importing the same dataset to work on
data_dummies = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Assignment(Association1)\\my_movies.csv",sep=",")
data_dummies = data_dummies.iloc[:,5:]

# applying apriori with support = 0.005, max length = 2

frequent_ori = apriori(data_dummies, min_support = 0.005, max_len=2, use_colnames = True)
frequent_ori.sort_values('support',ascending =False , inplace =True)

# Building rules with lift and minimum threshold =1
rules_ori = association_rules(frequent_ori, metric= 'lift', min_threshold =1)
rules_ori.sort_values('lift', ascending =False, inplace =True)

# Eliminating the reducdancy
def to_list_ori(i):
    return(sorted(i))
    
ori_add = rules_ori.antecedents.apply(to_list_ori) + rules_ori.consequents.apply(to_list_ori)

ori_add = ori_add.apply(sorted)

ori_set = list(ori_add)
unique_ori = [list(m) for m in set(tuple(i) for i in ori_set)]

index_ori = []
for i in unique_ori:
    index_ori.append(ori_set.index(i))

# rules without reducdancy    
rules_without_ori = rules_ori.iloc[index_ori,:]

# Sorting
rules_without_ori.sort_values('lift', ascending = False, inplace = True)

Support_ori = rules_without_ori["support"]
confidence_ori = rules_without_ori["confidence"]
lift_ori = rules_without_ori["lift"]

# plotting 3D for the rules generated
fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection = '3d')
ax1.scatter(Support_ori,confidence_ori,lift_ori)
ax1.set_xlabel("Support")
ax1.set_ylabel("Confidence")
ax1.set_zlabel("lift")

##Plotting the a scatter plot

plt.scatter(Support_ori,confidence_ori, c =lift_ori, cmap ='gray')
plt.colorbar()
plt.xlabel("Support")
plt.ylabel("confidence")


