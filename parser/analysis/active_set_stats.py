import fileinput
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    counts = {'train': [],
              'valid': [],
              'test': []}

    for line in fileinput.input():
        if line.startswith("[dynet]"):
            continue

        tr, va, te = line.split("\t")
        tr = np.fromstring(tr, dtype=np.uint32, sep=' ')
        va = np.fromstring(va, dtype=np.uint32, sep=' ')
        te = np.fromstring(te, dtype=np.uint32, sep=' ')

        tr = tr.reshape(-1, 3)
        va = va.reshape(-1, 3)
        te = te.reshape(-1, 3)

        counts['train'].append(tr)
        counts['valid'].append(va)
        counts['test'].append(te)

    histogram = False
    boxplot = True

    #  title = "English"
    #  fn = "en"

    #  title = "Vietnamese"
    #  fn = "vi"

    title = "Chinese"
    fn = "zh"

    params = dict(showfliers=False)

    if boxplot:
        plt.rc('text', usetex=True)
        # plt.rc('font', family='serif', serif='Times', size=14)
        plt.rc('font', size=16)
        plt.rc('text.latex', preamble=[
            r'\usepackage{arev}',
            r'\usepackage{amsmath}',
            r'\usepackage{times}',
            r'\usepackage{bm}',
            r'\newcommand{\norm}[1]{\left\lVert#1\right\rVert}'
            #  r'\newcommand{\norm}[1]{||#1||}'
            #  r'\newcommand{\norm}[1]{\lVert#1\rVert}'
        ])
        fig, ((ax1, ax2),
              #  (ax3, ax4),
              (ax5, ax6)) = plt.subplots(2, 2,
                                         sharex=True,
                                         sharey="row",
                                         figsize=(13, 4))

        #  fig.suptitle(title)
        data = [d[:, 0] for d in counts['train']]
        ax1.boxplot(data, **params)
        ax1.set_title("Training")
        # ax1.set_ylabel(r"$\norm{\bm{y}^\star}_0$", labelpad=6)
        ax1.set_ylabel(r"$\norm{\bm{p}^\star}_0$", labelpad=6)
        ax1.set_yticks((1, 25, 50))

        data = [d[:, 0] for d in counts['valid']]
        ax2.set_title("Validation")
        ax2.boxplot(data, **params)

        #  data = [d[:, 1] / d[:, 2] * 100 for d in counts['train']]
        #  ax3.boxplot(data, **params)
        #  ax3.set_ylabel("% nonzero arcs")

        #  data = [d[:, 1] / d[:, 2] * 100 for d in counts['valid']]
        #  ax4.boxplot(data, **params)

        data = [d[:, 1] / ((np.sqrt(d[:, 2]) - 1)) for d in counts['train']]
        #  ax5.set_ylabel(r"$\norm{\bm{u}^\star}_0 / n$", labelpad=12)
        ax5.set_ylabel(r"$\norm{\bm{\mu}^\star}_0 / n$", labelpad=12)
        ax5.boxplot(data, **params)
        ax5.set_xticks((1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50))
        ax5.set_xticklabels((1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50))
        ax5.set_yticks((1, 2, 3, 4, 5))
        ax5.set_xlabel("Epoch")

        data = [d[:, 1] / ((np.sqrt(d[:, 2]) - 1)) for d in counts['valid']]
        ax6.boxplot(data, **params)
        ax6.set_xticks((1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50))
        ax6.set_xticklabels((1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50))
        ax6.set_yticks((1, 2, 3, 4, 5, 6))
        ax6.set_xlabel("Epoch")

        fig.tight_layout()
        #  plt.show()
        #  plt.savefig(f"{fn}_stats.png")
        #plt.savefig(f"../../paper_icml/img/{fn}_stats.pdf")
        print('saving')
        plt.savefig(f"/home/vlad/slides-and-posters/2018-sparsemap-icml/{fn}_stats.pdf")
        exit()


    if histogram:
        bins = np.arange(1, 50)
        perc_bins = np.arange(0, 101)
        for name, data in counts.items():
            sz = len(data[0])
            for i in range(len(data)):
                fig, ax = plt.subplots()
                plt.title(f"{name}: trees@epoch {i}")
                ax.set_xlim(1, 50)
                ax.set_ylim(0, sz)
                ax.set_xlabel("#trees in active set")
                ax.set_ylabel("#sentences")
                plt.hist(data[i][:, 0], bins=bins)
                plt.savefig(f"animations/trees_{name}_{i:03d}.png")
                plt.close()

                fig, ax = plt.subplots()
                plt.title(f"{name}: arcs@epoch {i}")
                ax.set_xlim(0, 100)
                ax.set_ylim(0, sz)
                ax.set_xlabel("% nonzero arcs")
                ax.set_ylabel("#sentences")
                perc = data[i][:, 1] / data[i][:, 2] * 100
                plt.hist(perc, bins=perc_bins)
                plt.savefig(f"animations/arcs_{name}_{i:03d}.png")
                plt.close()
