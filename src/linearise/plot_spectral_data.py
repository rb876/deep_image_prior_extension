import os
import numpy as np 
import yaml
import matplotlib
import matplotlib.pyplot as plt 

def singular_values_plot(s, labels, colors, line):

    fig, ax = plt.subplots(figsize=(6, 6))
    for val, lab, col, lin in zip(s, labels, colors, line):
        len_vec = list(range(len(val)))
        ax.plot(len_vec, val, linestyle=lin, linewidth=2.5, color=col, label=lab)
    ax.grid()
    ax.legend()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('Singular Values', fontsize=16)
    ax.set_yscale('log')
    ax.set_xlabel('# projections', fontsize=12)
    ax.set_ylabel('Magnitude', fontsize=12)
    lgd = ax.legend(loc='upper right', ncol=2, framealpha=1.)
    fig.savefig('singular_values_comparison_plot.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')
    
def singular_vectors_plot(v1, v2, n_cols=4, plot_first_k=None, labels=None, filename=None, reorder_idx=None):
    from matplotlib.ticker import MaxNLocator
    from matplotlib.lines import Line2D

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    matplotlib.rcParams.update({'font.size': 16})
    vec1_len = list(range(v1.shape[1]))
    n_proj = v1.shape[0] if plot_first_k is None else plot_first_k
    n_rows = n_proj // n_cols
    n_rows += n_proj % n_cols   
    if reorder_idx is None:
        reorder_idx = slice(None)
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(25, 15),  facecolor='w', edgecolor='k', constrained_layout=True)
    axs = axs.flatten()
    for i in range(n_proj):
        fct = np.sign(np.dot(v1[i, :], v2[i, :]))
        axs[i].plot(vec1_len, v1[i, reorder_idx]*fct, '-', color='#EC2215', alpha=1, linewidth=2.5, label=labels[0])
        axs[i].plot(vec1_len, v2[i, reorder_idx], '-', color='#3D78B2', alpha=.75, linewidth=2.5, label=labels[1])
        axs[i].set_ylabel("$v_{%s}$" % str(i), fontsize=22)
        axs[i].spines['right'].set_visible(False)
        axs[i].spines['top'].set_visible(False)
        axs[i].yaxis.set_major_locator(MaxNLocator(5))
        axs[i].set_xlabel('parameters', fontsize=22)
    title = fig.suptitle("Spectral Comparison", fontsize=32)
    legend_elements = [Line2D([0], [0], color='#EC2215', alpha=1, linewidth=2.5, label=labels[0]),
                       Line2D([0], [0], color='#3D78B2', alpha=.75, linewidth=2.5, label=labels[1])]

    lgd = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.05), ncol=2, framealpha=1., fontsize=22)
    fig.savefig(filename, bbox_extra_artists=(lgd, title, ), bbox_inches='tight')

def proj_diff_params_plot(delta_params_list, v_list, labels, colors, filename):

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    matplotlib.rcParams.update({'font.size': 16})
    fig, ax = plt.subplots(figsize=(6, 6))
    title = ax.set_title('Projected Parameters Difference')
    for (delta_params, v, lab, col) in zip (delta_params_list, v_list, labels, colors):
        prj_error = []
        n_proj = v.shape[0]
        vec_len = list(range(n_proj))
        for i in range(n_proj):
            tmp = (np.dot(delta_params, v[i, :]) / np.linalg.norm(delta_params, ord=2))**2
            prj_error.append(tmp)
        if i == 0: 
            ax.plot(vec_len, prj_error, '-.', linewidth=2.5, color=col, label=lab)
        else: 
            ax2 = ax.twinx()
            ax2.plot(vec_len, prj_error, '-.', linewidth=2.5, color=col, label=lab)
    ax.grid()
    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.01), ncol=2, framealpha=1., fontsize=22)
    fig.savefig(filename, bbox_extra_artists=(lgd, title, ), bbox_inches='tight')

def singular_vectors_3D_plot(v_s, proj_idx, iters, n_rows, n_cols, colors, opacity, labels, iter_labels, filename, reorder_idx=None):
    from matplotlib.lines import Line2D

    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)
    matplotlib.rcParams.update({'font.size': 12})

    len_vec = np.arange(v_s[0].shape[1])
    if reorder_idx is None:
        reorder_idx = slice(None)
    fig = matplotlib.pyplot.Figure(figsize=(20, 8))
    for i, proj in enumerate(proj_idx):
        ax = fig.add_subplot(n_rows, n_cols, i+1,  projection='3d', facecolor='w')
        for k,  (v, it, cs, alpha) in enumerate(zip(v_s, iters, colors, opacity)):
            fct = np.sign(np.dot(v_s[0][proj, :], v_s[k][proj, :]))
            ax.plot(len_vec,  v[proj, reorder_idx]*fct, zs=it, zdir='y', color=cs, alpha=alpha, zorder=10-k, linewidth=0.5)
        ax.set_zlabel("$v_{%s}$" % str(proj+1), fontsize=16, labelpad=8. if proj >= 500 else 4.)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.locator_params(axis='x', nbins=5)
        ax.locator_params(axis='z', nbins=3)
        ax.xaxis.set_label_text('parameters' + ' (1e6)', fontsize=14)
        ax.xaxis.offsetText.set_visible(False)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12)        
        ax.set_yticks([np.mean(iters[:len(iters)//2]), np.mean(iters[len(iters)//2:])])
        ax.set_yticklabels(labels, rotation=90)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.set_aspect('auto')
        ax.grid(None)
    
    def compute_hoyer_score(x):
        sqrtN = np.sqrt(x.shape[0])
        return (sqrtN - (np.abs(x).sum()/np.sqrt((x**2).sum())) ) / (sqrtN - 1)

    def add_histogram(ax, slc, xlim_max=None):
        max_bin = 0.
        for v, it, cs in zip(v_s, iters, colors):
            counts, bins= np.histogram(np.abs(v[slc].flatten()), bins=1000)
            max_bin = max(bins[-1], max_bin)
            counts = [el if el > 0 else 0 for el in np.log10(counts)]
            ax.bar(bins[:-1], counts, zs=it, zdir='y', color=cs,  width=0.001, alpha=0.8)
        if xlim_max is None:
            xlim_max = max_bin
        for v, it, cs in zip(v_s, iters, colors):
            ave_hoyer_score = np.mean([compute_hoyer_score(el) for el in v[slc]])
            # ave_hoyer_score = np.mean([compute_hoyer_score(el) for el in v])
            ax.text(0.8*xlim_max, it-0.85, 0, '({:.2f})'.format(ave_hoyer_score), 'x', color=cs, fontsize=10)
        
        import matplotlib.ticker as mticker
        def log_tick_formatter(val, pos=None):
            return "{:2.0e}".format(10**val).replace("+0", "")

        ax.zaxis.set_major_formatter(mticker.FuncFormatter(log_tick_formatter))
        ax.locator_params(axis='z', nbins=3)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=12)
        ax.zaxis.set_tick_params(labelsize=12) 
        ax.set_xlim(0, xlim_max)
        ax.locator_params(axis='x', nbins=5)      
        ax.set_yticks([np.mean(iters[:len(iters)//2]), np.mean(iters[len(iters)//2:])])
        ax.set_yticklabels(labels, rotation=90)
        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.grid(None)
    
    ax = fig.add_subplot(n_rows, n_cols, i+2, projection='3d', facecolor='w')
    add_histogram(ax, slice(0, 20), xlim_max=0.375)
    ax.set_title('Histogram (a)', y=0.95)
    ax = fig.add_subplot(n_rows, n_cols, i+3, projection='3d', facecolor='w')
    add_histogram(ax, slice(975, 995), xlim_max=0.375)
    ax.set_title('Histogram (b)', y=0.95)

    fig.subplots_adjust(top=0.95)

    legend_elements = [Line2D([0], [0], color=colors[i], alpha=1, linewidth=2, label=el) for i, el in enumerate(iter_labels)]
    lgd = fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.04), ncol=len(iter_labels), framealpha=1., fontsize=14)
    fig.savefig(filename,  bbox_inches='tight')
    
def main(): 

    with open(os.path.join(os.path.dirname(__file__), 'runs.yaml'), 'r') as f:
        runs = yaml.load(f, Loader=yaml.FullLoader)

    reorder_idx = np.load(os.path.join(os.path.dirname(__file__), 'reorder_idx.npy'))

    s, v, params = [], [], []
    for key in runs.keys():
        for _, path in runs[key].items():
            data = np.load(path)
            s.append(data['values'])
            v.append(data['vectors'])
            params.append(data['params'])
    
    singular_vectors_3D_plot(
        v,
        proj_idx=[
            0, 1, 2, 3, 499, 994],
        iters=[1, 3, 5, 8, 10, 12],
        n_rows=2,
        n_cols=4,
        colors=['#084B8D', '#3D78B2', '#96D6BA', '#084B8D', '#3D78B2', '#96D6BA'],
        opacity=[1, 1, 1, 1, 1, 1],
        labels=['EDIP', 'DIP'],
        iter_labels=['$\\theta^{\\mathrm{conv}}$', '$\\theta^{[100]}$', '$\\theta^{\\mathrm{init}}$'],
        filename='test.pdf',
        reorder_idx=reorder_idx
        )

    singular_values_plot(
        s, 
        labels=['EDIP ($\\theta^{\\mathrm{conv}}$)', 'EDIP ($\\theta^{[100]}$)', 'EDIP ($\\theta^{\\mathrm{init}}$)',
                'DIP ($\\theta^{\\mathrm{conv}}$)', 'DIP ($\\theta^{[100]}$)', 'DIP ($\\theta^{\\mathrm{init}}$)'],
        colors=['#084B8D', '#3D78B2', '#96D6BA', '#084B8D', '#3D78B2', '#96D6BA'],
        line=['-', '-', '-', (0, (2, 5)), (0, (2, 5)), (0, (2, 5))]
        )
    
if __name__ == "__main__":
    main()
