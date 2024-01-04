import pandas as pd
import numpy as np

data = pd.DataFrame(data=pd.read_csv('Human.csv'))
person = pd.DataFrame(data=pd.read_csv('Person.csv'))

n_male = data['Gender'][data['Gender']=='male'].count()
n_female = data['Gender'][data['Gender']=='female'].count()
total_ppl = data['Gender'].count()

p_male = n_male/total_ppl
p_female = n_female/total_ppl

data_means = data.groupby('Gender').mean()
data__variance = data.groupby('Gender').var()

male_Height_mean = data_means['Height'][data__variance.index=='male'].values[0]
male_Weight_mean = data_means['Weight'][data__variance.index=='male'].values[0]
male_footsize_mean = data_means['Foot_Size'][data__variance.index=='male'].values[0]

male_Height_variance = data__variance['Height'][data__variance.index=='male'].values[0]
male_Weight_variance = data__variance['Weight'][data__variance.index=='male'].values[0]
male_footsize_variance = data__variance['Foot_Size'][data__variance.index=='male'].values[0]

female_Height_mean = data_means['Height'][data__variance.index=='female'].values[0]
female_Weight_mean = data_means['Weight'][data__variance.index=='female'].values[0]
female_footsize_mean = data_means['Foot_Size'][data__variance.index=='female'].values[0]

female_Height_variance = data__variance['Height'][data__variance.index=='female'].values[0]
female_Weight_variance = data__variance['Weight'][data__variance.index=='female'].values[0]
female_footsize_variance = data__variance['Foot_Size'][data__variance.index=='female'].values[0]

def p_x_given_y(x,mean_y,variance_y):
    p = 1/(np.sqrt(2*np.pi*variance_y))*np.exp((-(x-mean_y)**2)/(2*variance_y))
    return p

PMale = p_male*p_x_given_y(person['Height'][0],male_Height_mean,male_Height_variance)*p_x_given_y(person['Weight'][0],male_Weight_mean,male_Weight_variance)*p_x_given_y(person['Foot_Size'][0],male_footsize_mean,male_footsize_variance)     
PFemale = p_female*p_x_given_y(person['Height'][0],female_Height_mean,female_Height_variance)*p_x_given_y(person['Weight'][0],female_Weight_mean,female_Weight_variance)*p_x_given_y(person['Foot_Size'][0],female_footsize_mean,female_footsize_variance)

if(PMale>PFemale):
     print("the given data belongs to male with probability of",PMale)
else:
     print("the given data belongs to female with probability of",PFemale)   

