import numpy as np 
import yaml
import matplotlib
import matplotlib.pyplot as plt 

def singular_values_comparison_plot(s, labels, colors, line):


    fig, ax = plt.subplots(figsize=(6, 6))
    for val, lab, col, lin in zip(s, labels, colors, line):
        len_vec = list(range(len(val)))
        ax.plot(len_vec, val, lin, linewidth=2.5, color=col, label=lab)
    ax.grid()
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Singular Values Comparison', fontsize=16)
    ax.set_yscale('log')
    ax.set_xlabel('# projections', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    lgd = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, framealpha=1.)
    fig.savefig('singular_values_comparison_plot.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    


def singular_vecors_comparison_plot(v1, v2, n_cols=4, plot_first_k=None, labels=None, filename=None):
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    matplotlib.rcParams.update({'font.size': 16})

    vec1_len = list(range(v1.shape[1]))
    n_proj = v1.shape[0] if plot_first_k is None else plot_first_k
    n_rows = n_proj // n_cols
    n_rows += n_proj % n_cols   
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(25, 15),  facecolor='w', edgecolor='k', constrained_layout=True)
    axs = axs.flatten()
    for i in range(n_proj):
        fct = np.sign(np.dot(v1[i, :], v2[i, :]))
        l1 = axs[i].plot(vec1_len, v1[i, :]*fct, '-', color='#EC2215', alpha=1, linewidth=2.5, label=labels[0])
        l2 = axs[i].plot(vec1_len, v2[i, :], '-', color='#3D78B2', alpha=.75, linewidth=2.5, label=labels[1])
        axs[i].set_ylabel("$v_{%s}$" % str(i), fontsize=22)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].yaxis.set_major_locator(MaxNLocator(5))
        axs[i].set_xlabel('parameters', fontsize=22)
    title = fig.suptitle("Spectral Comparison", fontsize=32)
    legend_elements = [Line2D([0], [0], color='#EC2215', alpha=1, linewidth=2.5, label=labels[0]),
                       Line2D([0], [0], color='#3D78B2', alpha=.75, linewidth=2.5, label=labels[1])]

    lgd = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, framealpha=1., fontsize=22)
    fig.savefig(filename, bbox_extra_artists=(lgd, title, ), bbox_inches='tight')
    

# def projected_diff_params_plot(delta_params, v):

#     n_proj = v.shape[0]
#     prj_error = []
#     for i in range(n_proj):
#         prj_error.append(np.abs(np.dot(delta_params, v[i, :])))
    
#     fig, ax = plt.subplots(figsize=(6, 6))
#     ax.plot(list(range(n_proj)), prj_error, '-.', linewidth=2.5, color='green')
#     ax.grid()
#     fig.savefig('projected_error_plot.pdf')

with open('./runs.yaml', 'r') as f:
    runs = yaml.load(f, Loader=yaml.FullLoader)

s, v, params = [], [], []
for key in runs.keys():
    for _, path in runs[key].items():
        data = np.load(path)
        s.append(data['values'])
        v.append(data['vectors'])
        params.append(data['params'])

singular_values_comparison_plot(s, ['Init (pretrained)', 'Finetuned (starting pretrained)', 'Init (random)', 'Finetuned (starting random)'], ['#EC2215', '#EC2215', '#3D78B2', '#3D78B2'], ['-', '--', '-', '--'])
singular_vecors_comparison_plot(v[0], v[1], plot_first_k=16, labels=['Init (pretrained)', 'Finetuned (starting pretrained)'],  filename='singular_vectors_comparison_plot_1.pdf')
singular_vecors_comparison_plot(v[2], v[3], plot_first_k=16, labels=['Init (random)', 'Finetuned (starting random)'], filename='singular_vectors_comparison_plot_2.pdf')
singular_vecors_comparison_plot(v[0], v[3], plot_first_k=16, labels=['Init (pretrained)', 'Finetuned (starting random)'], filename='singular_vectors_comparison_plot_3.pdf')
singular_vecors_comparison_plot(v[1], v[3], plot_first_k=16, labels=['Finetuned (starting pretrained)', 'Finetuned (starting random)'], filename='singular_vectors_comparison_plot_4.pdf')
