import sys
import os
import matplotlib.pyplot as plt

def get_value(s, key, offset):
    i = s.index(key, offset)
    return s[s.index(" ", i + len(key)):s.index("\n", i)]


if __name__ == '__main__':

    lang = sys.argv[1]

    fnames = {
        'hinge': f"{lang}_map_aug.txt",
        'perc': f"{lang}_map_per.txt",
        'sparsemap-hinge': f"{lang}_spa_aug.txt",
        'sparsemap-perc': f"{lang}_spa_per.txt",
    }

    fig_ep_tr = plt.figure()
    fig_ep_va = plt.figure()
    fig_ts_tr = plt.figure()
    fig_ts_va = plt.figure()
    ax_ep_tr = fig_ep_tr.gca()
    ax_ep_va = fig_ep_va.gca()
    ax_ts_tr = fig_ts_tr.gca()
    ax_ts_va = fig_ts_va.gca()

    for key, fn in fnames.items():

        print(key)
        with open(os.path.join("results", fn)) as f:
            fc = f.read()

        epochs = []
        losses = []
        train_scores = []
        valid_scores = []
        timestamps = []

        i = 0
        while True:
            try:
                i = fc.index("epoch", i + 1)
            except ValueError:
                break

            epoch = int(fc[fc.index(" ", i):fc.index("\n", i)])

            loss = float(get_value(fc, "train loss", i))
            train_uas = float(get_value(fc, "train UAS", i))
            valid_uas = float(get_value(fc, "valid UAS", i))
            elapsed = float(get_value(fc, "elapsed", i))

            epochs.append(epoch)
            losses.append(loss)
            train_scores.append(train_uas)
            valid_scores.append(valid_uas)
            timestamps.append(elapsed)

        ax_ep_tr.plot(epochs, train_scores, label=key, marker='o')
        ax_ep_va.plot(epochs, valid_scores, label=key, marker='o')
        ax_ts_tr.plot(timestamps, train_scores, label=key, marker='o')
        ax_ts_va.plot(timestamps, valid_scores, label=key, marker='o')

    ax_ep_tr.set_title(lang)
    ax_ts_tr.set_title(lang)
    ax_ep_va.set_title(lang)
    ax_ts_va.set_title(lang)

    ax_ep_tr.set_ylabel("train UAS")
    ax_ep_tr.set_xlabel("epoch")

    ax_ts_tr.set_ylabel("train UAS")
    ax_ts_tr.set_xlabel("time (s)")

    ax_ep_va.set_ylabel("valid UAS")
    ax_ep_va.set_xlabel("epoch")

    ax_ts_va.set_ylabel("valid UAS")
    ax_ts_va.set_xlabel("time (s)")

    ax_ep_tr.legend()
    ax_ep_va.legend()
    ax_ts_tr.legend()
    ax_ts_va.legend()

    x = "png"
    fig_ep_tr.savefig(os.path.join("results", "fig", f"{lang}_train_epoch.{x}"))
    fig_ep_va.savefig(os.path.join("results", "fig", f"{lang}_valid_epoch.{x}"))
    fig_ts_tr.savefig(os.path.join("results", "fig", f"{lang}_train_time.{x}"))
    fig_ts_va.savefig(os.path.join("results", "fig", f"{lang}_valid_time.{x}"))



