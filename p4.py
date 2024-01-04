import pandas as pd 
df_tennis = pd.DataFrame(data = pd.read_csv('PlayTennisTrain.csv'))
print(df_tennis)
def entropy(probs):
    import math
    return sum([-prob*math.log(prob,2) for prob in probs])

def entropy_of_list(a_list):
    from collections import Counter
    cnt = Counter(x for x in a_list)
    print("no and yes class:",a_list.name,cnt)
    num_instances = len(a_list)*1.0
    probs = [x/num_instances for x in cnt.values()]
    return entropy(probs)
print(df_tennis['playtennis'])
total_entropy = entropy_of_list(df_tennis['playtennis'])
print("entropy of given playtennis datasets:",total_entropy)    

def information_gain(df,split_attribute_name,target_attribute_name,trace=0):
    print("info gain calculation of",split_attribute_name)
    df_split = df.groupby(split_attribute_name)
    for name,group in df_split:
        print(name)
        print(group)
    nobs = len(df.index)*1.0
    df_agg1 = df_split.agg({target_attribute_name:lambda x:entropy_of_list(x)})
    df_agg2 = df_split.agg({target_attribute_name:lambda x:len(x)/nobs})
    df_agg1.columns = ['entropy']
    df_agg2.columns = ['proportion']
    new_entropy = sum(df_agg1['entropy']*df_agg2['proportion'])
    old_entropy = entropy_of_list(df[target_attribute_name])
    return old_entropy-new_entropy
print("info gain for outlook is:"+str(information_gain(df_tennis,'outlook','playtennis')),"\n")
print("info gain for humidity is:"+str(information_gain(df_tennis,'humidity','playtennis')),"\n")  
print("info gain for temperature is:"+str(information_gain(df_tennis,'temperature','playtennis')),"\n")

def id3(df,target_attribute_name,attribute_name,default_class='None'):
    from collections import Counter
    cnt = Counter(x for x in df[target_attribute_name])
    if len(cnt)==1:
        return next(iter(cnt))
    elif df.empty or(not attribute_names):
        return default_class
    else:
        default_class = max(cnt.keys())
        gainz = [information_gain(df,attr,target_attribute_name) for attr in attribute_names]
        index_of_max = gainz.index(max(gainz))
        best_attr = attribute_names[index_of_max]
        tree  = {best_attr:{}}
        remaining_attribute_names = [i for i in attribute_name if i!=best_attr]
    for attr_val,data_subset in df.groupby(best_attr):
        subtree = id3(data_subset,target_attribute_name,remaining_attribute_names,default_class)
        tree[best_attr][attr_val]=subtree
    return tree

attribute_names = list(df_tennis.columns)
print("list of attributes:",attribute_names)
attribute_names.remove('playtennis')
print("pridicting attributes:",attribute_names)
from pprint import pprint
tree = id3(df_tennis,"playtennis",attribute_names)
pprint("\n the result decision tree is:\n")
pprint(tree)                    
