import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
sns.set(font_scale = 1.5)
exp1 = False
if exp1:
    data = pd.read_csv("new_base_unity_compare.csv") # base - unity , experiment 1 from paper
else: # exp2
    data = pd.read_csv("new_unity_auto_compare.csv") # unity - auto , this is the opposite of what it says in the paper but appears the same
print(data)

# d = {'color': ['C0', 'k'], "ls" : ["-","--"]}
# plfig = sns.displot(data,x = data.Value, hue = data.Reasoner, kind = "kde" ,bw_adjust = 0.2, legend=False)
# for ax in plfig.axes.flat:
#     ax.lines[0].set_linestyle("--")
#     ax.lines[0].set_linewidth(5)
#     ax.lines[1].set_linewidth(5)
# a = Line2D([], [], color="blue", label="Standard reasoner")
# b = Line2D([], [], color="orange", linestyle="--" ,label="Guided reasoner")

# plt.legend(handles=[a, b], fontsize = 24)

plt.ticklabel_format(style='plain', axis='x')
# plt.fig.suptitle("Nodes Explored")
plt.ticklabel_format(style='plain', axis='y')
pl = sns.barplot(x = data["Query"],y = data["Change"],order = data.sort_values("Change").Query)
pl.set_yscale("symlog")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False)
#plt.xticks(rotation = 90)
# for container in pl.containers:
#     pl.bar_label(container)
#plt.xlabel("Change")
plt.gcf().set_size_inches(10,10)
sns.set_style("ticks")
plt.rcParams.update({'font.size': 20})
plt.xlabel('Queries')
plt.ylabel('Difference (log)')
#plt.ylabel("Change")
#plt.suptitle("Change in nodes explored between \n autoencoder and unification embedding guided reasoners")
# hatches = itertools.cycle(['/', '//', '+', '-', 'x', '\\', '*', 'o', 'O', '.'])
# for i, bar in enumerate(pl.patches):
#     if i % 7 == 0:
#         hatch = next(hatches)
#     bar.set_hatch(hatch)
plt.tight_layout()
# pl.legend()
plt.show()