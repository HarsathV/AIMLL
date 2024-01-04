import pandas as pd
import numpy as np

data = pd.DataFrame(data=pd.read_csv(Human.csv'))
person = pd.DataFrame(data=pd.read_csv('Person.csv'))

n_male = data[ Gender" ][data[ Gender" ]
n_female = data[ Gender" ][data[ Gender]
total_ppl = data[ Gender" ].count()

‘male"]. count ()
female". count()

P_male = n_male/total_ppl
P_female = n_female/total_ppl

data_means = data.groupby( Gender" ).mean()
data_variance = data.groupby(’Gender').var()

male height mean = data_means['Height'][data_variance. index
‘male’ ].values[6]

nale_weight mean = data_means['Weight'][data_variance. index
‘male’ ].values[6]

nale_footsize_mean = data_means['Foot_Size'][data_variance. index
== 'male’].values[e]

nale_height_variance =
data_variance[ ‘Height ][data_variance. index
nale weight variance =
data_variance[ ‘Weight ][data_variance. index
male_footsize_variance =

data_variance[ 'Foot_Size'][data_variance. index
‘male’ ].values[0]

‘male’ ].values[6]

‘male’ ].values[6]

female_height mean = data_means[ 'Height'][data_variance. index ==
female" ].values[0]

female_weight mean = data_means[ 'Weight'][data_variance. index ==
‘female'].values[@]

female_footsize mean =
data_means['Foot_Size'][data_variance. index
*female'].values[o]

female_height variance =
data_variance[ ‘Height ][data_variance. index ==
*fenale'].values[e]

female_weight_variance =
data_variance[ ‘Weight ][data_variance. index
‘fenale'].values[e]

female_footsize variance =
data_variance( 'Foot_Size'][data_variance. index
*female'].values[0]

def p_x_given_y(x, mean_y, variance_y):
# Input the arguments into a probability density function
p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-
mean_y)**2)/(2*variance_y))
return p

PMale = P_male * p_x_given_y(person[ 'Height'][e],
nale_height_mean, male height variance) *
p_x_given_y(person[ ‘Weight '][6], male_weight mean,

male weight variance) * p_x_given_y(person['Foot_Size'][e],
male_footsize mean, male_footsize variance)

PFemale = P_female * p_x_given_y(person[‘Height'][@],
female_height mean, female_height_variance) *
p_x_given_y(person[ ‘Weight"][0], female weight mean,
Female_weight_variance) * p_x_given_y(person['Foot_Size'][e],
female_footsize_mean, female_footsize variance)

if(PMale > PFemale):
print("The given data belongs to Male with Probability of
",PMale)
else:
print,
",PFemale)

“The given data belongs to Female with Probability of
