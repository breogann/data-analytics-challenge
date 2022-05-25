#%% Import  modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from functions import *

#%% Import data

raw_df = pd.read_excel('../data/measurements2.xlsx')

#%% Clean data

df = raw_df.copy()

# Drop two last columns
df = df.iloc[:, :-2]

# Drop `specials` column
df = df.drop('specials', axis = 1)

# Create dummies for `gas_type`
df = pd.get_dummies(df, columns = ['gas_type'])

# Apply forward fill to `temp_inside`
df['temp_inside'] = df['temp_inside'].fillna(method = 'ffill')

# Create columns for speed squared
df['speed2'] = df['speed'] ** 2

# Set y and X

y, X = df['consume'], df.drop(columns = 'consume')

# %% Check correlations

corr(df)

#%% Check non linear test

non_linear_test(y, X)

# %% OLS test

ols_analysis(y, X)

# %% RFR analysis

rfr_analysis(y, X)

#%% Tunned OLS models

models = {
    'speed_simple' : ['speed',
            'gas_type_E10', 
            'gas_type_SP98',
            ],
    'speed_squared' : ['speed',
            'speed2',
            'gas_type_E10', 
            'gas_type_SP98',
            ],
    'temp_inside' : ['speed',
            'speed2',
            'gas_type_E10', 
            'gas_type_SP98',
            'temp_inside'],
    'rain' : ['speed',
            'speed2',
            'gas_type_E10', 
            'gas_type_SP98',
            'temp_inside',
            'rain']}

models_dic = {} # Model name, p-value of hypothesis that consume is equal for each carburant

# Compute all models, obtain results and save them in a dictionary
for name, indep_vars in models.items():
    dep_var = 'consume'
    indep_var = indep_vars

    reg = clean_reg(dep_var, indep_var, df,
                    intercept = True)
    print(reg.t_test('gas_type_E10 = gas_type_SP98').pvalue)
    print(reg.summary())

    models_dic[name] = {
                        'E10' : round(reg.params['gas_type_E10'], 3),
                        'SP98' : round(reg.params['gas_type_SP98'], 3),
                        'pvalue' : round(float(reg.t_test('gas_type_E10 = gas_type_SP98').pvalue), 2)
    }

#%% Plot different models

models_df = pd.DataFrame(models_dic)
pvalues = models_df.loc['pvalue']

hyp_pvalues = sns.barplot(x = models_df.columns, y = pvalues, color = '#006699')

# Add a horizontal line at 0.05
hyp_pvalues.axhline(y = 0.05, color = 'r', linestyle = '--')
hyp_pvalues.set_title('P-values of hypothesis that E10 and SP98 are equal')
hyp_pvalues.set_xlabel('Model')
hyp_pvalues.set_ylabel('P-value')


# %% Visualize results mean spead vs consumption

# Create a scatterplot with Seaborn to visualize the relationship between the
# mean speed and consumption.

sns.scatterplot( x = df['speed'], y = df['consume'],
                hue = df['gas_type_E10'],)

# %%
