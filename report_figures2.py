# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
temp_vs_all = pd.read_excel(
    './numerical_comparison.xlsx', sheet_name='TMCMC_vs_all',
    header=0
)

index_names = temp_vs_all[temp_vs_all['Method']=='Bound'].index
temp_vs_all.drop(index_names, inplace=True)

plot_dim = [5, 10, 30, 50]
fig, axes = plt.subplots(2, 2, figsize=(6.5,5), tight_layout=True, sharex=True)

for i,n in enumerate(plot_dim):
    ax = axes.flatten()[i]
    data = temp_vs_all[temp_vs_all['Dimension']==n]

    g1 = sns.boxplot(data=data, x='Method', y='Result',
                     width=0.3, ax=ax,
                     meanline=True, showmeans=True, whis=(0,100))
    # g2 = sns.swarmplot(data=temp_vs_all, x='Method', y='Result', color='gray', 
    #                    ax=ax, **dict(marker='o'))
    ax.axhline(data['Benchmark'].iloc[0], linestyle='-.', color='tab:red')
    ax.set_ylabel('Risk', fontsize=9)
    ax.set_xlabel('Method', fontsize=9)
    ax.tick_params(axis='both', labelsize=9)
# fig.savefig('./figures/temp_vs_all.png', dpi=600)

# %%
temp_vs_mc = pd.read_excel(
    './numerical_comparison.xlsx', sheet_name='TMCMC_vs_MC',
    header=0
)

index_names = temp_vs_mc[temp_vs_mc['Method']=='bound'].index
temp_vs_mc.drop(index_names, inplace=True)

fig, ax = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
g1 = sns.boxplot(data=temp_vs_mc, x='Method', y='Result', width=0.4, ax=ax,
                 meanline=True, showmeans=True, whis=(0,100))
g2 = sns.swarmplot(data=temp_vs_mc, x='Method', y='Result', color='gray', 
                   ax=ax, **dict(marker='o'))
ax.axhline(temp_vs_mc['Benchmark'].iloc[0], linestyle='-.', color='tab:red')
ax.set_ylabel('Risk', fontsize=9)
ax.set_xlabel('Method', fontsize=9)
ax.tick_params(axis='both', labelsize=9)
fig.savefig('./figures/temp_vs_MC.png', dpi=600)

# %%
import numpy as np
import pandas as pd

folder = 'tmp_2023-08-15_07_05'

damage_data = np.load(f'./assets/{folder}/damage_condition.npz')
unique_condition, counts = damage_data['unique_condition'], damage_data['counts']

top = 3
sort_indx = np.argsort(counts)
top_condition = unique_condition[sort_indx[-top:], :]
top_rank, top_bridge = np.nonzero(top_condition)
top_rank += 1
top_bridge += 1

bridge_df = pd.DataFrame({'Rank': top_rank, 'Bridge': top_bridge})
bridge_df.to_csv(f'./assets/{folder}/top{top}_bridges.csv')

# %%
