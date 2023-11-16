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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

for dim in [30, 50, 100]:
    for useful in [3, 5, 10]:
        temp_vs_mc = pd.read_excel(
            './assets/numerical_comparison_AZ.xlsx', sheet_name='TMCMC_vs_MC',
            header=0 )
        index_names = temp_vs_mc[temp_vs_mc['Method']=='Bound'].index
        temp_vs_mc.drop(index_names, inplace=True)
        temp_vs_mc = temp_vs_mc[(temp_vs_mc['Dimension']==dim) & (temp_vs_mc['Useful dim.']==useful)]
        fig, ax = plt.subplots(1,1, figsize=(4,3), tight_layout=True)
        g1 = sns.boxplot(data=temp_vs_mc, x='Method', y='Result', width=0.4, ax=ax,
                        meanline=True, showmeans=True, whis=(0,100))
        g2 = sns.swarmplot(data=temp_vs_mc, x='Method', y='Result', color='gray', 
                        ax=ax, **dict(marker='o'))
        ax.axhline(temp_vs_mc['Benchmark'].iloc[0], linestyle='-.', color='tab:red')
        ax.set_ylabel('Risk', fontsize=9)
        ax.set_xlabel('Method', fontsize=9)
        ax.tick_params(axis='both', labelsize=9)
        fig.savefig(f'./assets/plots/{dim}_bridges_{useful}_useful_v2.png', dpi=600)

# %%
import numpy as np
import pandas as pd
import seaborn as sns


folder = 'tmp_2023-10-27_23_11'

damage_condition = np.load(f'./assets/{folder}/damage_condition.pkl', allow_pickle=True)

unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)

top = 3
sort_indx = np.argsort(counts)
top_condition = unique_condition[sort_indx[-top:], :]
top_rank, top_bridge = np.nonzero(top_condition)
top_rank += 1
top_bridge += 1

#bridge_df = pd.DataFrame({'Rank': top_rank, 'Bridge': top_bridge})
#bridge_df.to_csv(f'./assets/{folder}/top{top}_bridges.csv')

all_bridges = np.arange(1, max(top_bridge)+1)
all_ranks = np.full(all_bridges.shape, 99)

for bridge in all_bridges:
    mask = (top_bridge == bridge)
    if np.any(mask):
        min_rank = np.min(top_rank[mask])
        all_ranks[bridge-1] = min_rank

bridge_df = pd.DataFrame({'Bridge': all_bridges, 'Rank_v2': all_ranks})
bridge_df.to_csv(f'./assets/{folder}/top{top}_bridgesv2.csv')

#%%
import numpy as np
import pandas as pd
from UQpy.distributions import Uniform

# Set the folder name
folder = 'tmp_2023-11-15_10_16_rand_3'

# Load the damage condition data
damage_condition = np.load(f'./assets/{folder}/damage_condition.pkl', allow_pickle=True)
beta_array = np.load(f'./assets/{folder}/beta_array.pkl', allow_pickle=True)

# Identify the top damage conditions and ranks
unique_condition, counts = np.unique(damage_condition, axis=0, return_counts=True)
top = 3
sort_idx = np.argsort(counts)
top_condition = unique_condition[sort_idx[-top:], :]
top_rank, top_bridge = np.nonzero(top_condition)
top_rank += 1
top_bridge += 1

# Initialize a DataFrame for bridge data
bridge_data = pd.DataFrame({'Bridge ID': np.arange(1, len(beta_array)+1)})

# Initialize columns for ranks and set default rank to 99
for i in range(top):
    bridge_data[f'In Rank {i + 1}'] = 0
bridge_data['Rank'] = 99

# Fill in the rank information
for i in range(len(top_rank)):
    bridge_id = top_bridge[i]
    rank = top_rank[i]
    bridge_data.at[bridge_id - 1, f'In Rank {rank}'] = 1
    if rank < bridge_data.at[bridge_id - 1, 'Rank']:
        bridge_data.at[bridge_id - 1, 'Rank'] = rank

# Add the beta values
bridge_data['Beta'] = beta_array

# # Save the bridge data to a CSV file
# bridge_data.to_csv(f'./assets/{folder}/bridge_data.csv', index=False)

# Save the bridge data to an Excel file
with pd.ExcelWriter(f'./assets/{folder}/bridge_data.xlsx') as writer:
    bridge_data.to_excel(writer, sheet_name='Bridge Data', index=False)


# %%
# Function to calculate to extract results
def extract_results(All_results, method, n_br, n_use, idx=0):
    value = [result[method][idx] for result in All_results.values() if (result['input'][0] == n_br) and (result['input'][-1] == n_use)]
    return value

# Function to calculate the benchmark
def get_benchmark(All_results, n_br, n_use):
    benchmark = [result['benchmark'] for result in All_results.values() if (result['input'][0] == n_br) and (result['input'][-1] == n_use)]
    return benchmark

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def boxplots(dictionary, n_br, n_use):

    # Create a figure and axis
    fig, ax = plt.subplots(1,1,figsize=(4,3), tight_layout=True)

    # Create boxplots for TMCMC and MC columns for 30 br and 3 useful br
    TMCMC = extract_results(dictionary, 'TMCMC', n_br, n_use)
    MC = extract_results(dictionary, 'MC', n_br, n_use)
    # RB = extract_results(dictionary, 'RB', n_br, n_use)
    # RB_mean, RB_std = np.mean(RB), np.std(RB)
    TMCMC_mean, MC_mean = np.mean(TMCMC), np.mean(MC)
    TMCMC_std, MC_std = np.std(TMCMC), np.std(MC)
    TMCMC_min, TMCMC_max = np.min(TMCMC), np.max(TMCMC)
    MC_min, MC_max = np.min(MC), np.max(MC)
    both = pd.DataFrame({'TMCMC': TMCMC, 'MC': MC})
    precise = get_benchmark(dictionary, n_br, n_use)
    f1 = sns.boxplot(data=both, ax=ax,whis=(0,100), meanline=True, showmeans=True, width=0.4)
    

    # Add a horizontal line for Precise
    ax.axhline(y=precise[0], linestyle='-.', color='tab:red')

    # Add points for each row and label TMCMC and MC
    # sns.stripplot(data=both, color="orange", ax=ax)
    sns.swarmplot(data=both, palette=['gray','gray'], ax=ax, marker='o', size=4)
    
    # Set labels and title for the main plot
    ax.set_xlabel("Method",fontsize=9)
    ax.set_ylabel("Risk",fontsize=9)
    ax.tick_params(axis='both', labelsize=9)
    #ax.set_title("For {} useful and {} total bridges".format(n_use, n_br))

    # Show the plot
    plt.savefig('./assets/plots/{}_bridges_{}_useful.png'.format(n_br, n_use), dpi=600)
    plt.show()
    print(f'mean TMCMC = {TMCMC_mean}, mean MC = {MC_mean}')
    print(f'std TMCMC = {TMCMC_std}, std MC = {MC_std}')
    print(f'benchmark = {precise[0]}')
    print(f'min TMCMC = {TMCMC_min}, min MC = {MC_min}')
    print(f'max TMCMC = {TMCMC_max}, max MC = {MC_max}')
# %%
import pickle
file = './assets/Case_II_results.pickle'
with open(file, 'rb') as f:
    TM_vs_MC = pickle.load(f)

n_brs = [30, 50, 100]
useful_brs = [3, 5, 10]
for n_br in n_brs:
    for n_use in useful_brs:
        boxplots(TM_vs_MC, n_br, n_use)

# %%
import pickle
file = './assets/Case_II_results_100.pickle'
with open(file, 'rb') as f:
    TM_vs_MC_100 = pickle.load(f)

n_brs = [100]
useful_brs = [3, 5, 10]
for n_br in n_brs:
    for n_use in useful_brs:
        boxplots(TM_vs_MC_100, n_br, n_use)


# %%
# import the computational graph and plot it using networkx kawanda kawai layout
import networkx as nx
import matplotlib.pyplot as plt

G = nx.read_graphml('./assets/or_hw_comp.graphml')

# change graph attributes type
G_comp = nx.DiGraph()
for u, v, data in G.edges(data=True):
    data['length'] = float(data['length'])
    data['capacity'] = float(data['capacity'])
    data['lane'] = int(data['lane'])
    data['speed'] = float(data['speed'])
    data['bridge_id'] = int(data['bridge_id'])
    G_comp.add_edge(u, v, **data)
for n, data in G.nodes(data=True):
    G_comp.add_node(n, **data)
mapping = {n: int(n) for n in G_comp.nodes}
G_comp = nx.relabel_nodes(G_comp, mapping)
G_clean = G_comp.copy()
G_clean.remove_edge(38789281, 38789281)
#pos = nx.kamada_kawai_layout(G_comp, weight='length')
#%%
# save pos dictionary to a pickle file
import pickle

pos = pickle.load(open('./assets/plots/positions.npz', 'rb'))

nx.draw(G_comp, pos, node_size=5, width=0.5, node_color='blue', edge_color='gray', 
    with_labels=False, arrows=False, alpha=0.5)

# plt.savefig('./assets/plots/computational_graph.png', dpi=600)
# plt.savefig('./assets/plots/computational_graph.pdf', dpi=600)

# %%

pos_spectral = nx.spectral_layout(G_comp, weight='length')
pos_spring = nx.spring_layout(G_comp, weight='length')
pos_shell = nx.shell_layout(G_comp)
pos_random = nx.random_layout(G_comp)



# %%
fig, ax = plt.subplots(1,1,figsize=(8,6), tight_layout=True)
nx.draw(G_comp, pos=pos_spectral, node_size=5, width=0.5, node_color='blue', edge_color='gray', 
    with_labels=False, arrows=False, alpha=0.8,)
plt.savefig('./assets/plots/comp_graph_spectral.png', dpi=600)
# %%
fig, ax = plt.subplots(1,1,figsize=(8,6), tight_layout=True)
nx.draw(G_comp, pos, node_size=5, width=0.5, node_color='blue', edge_color='gray', 
    with_labels=False, arrows=False, alpha=0.8,)
plt.savefig('./assets/plots/comp_graph_kawada.png', dpi=600)
# %%
# draw the graph with specific edge drawn in red
fig, ax = plt.subplots(1,1,figsize=(8,6), tight_layout=True)
nx.draw(G_clean, pos_spectral, node_size=5, width=0.5, node_color='blue', edge_color='gray', 
    with_labels=False, arrows=False, alpha=0.5)
nx.draw_networkx_edges(G_comp, pos_spectral, edgelist=[(38789281, 38789281)], edge_color='red', width=1)

# %%
