import subprocess
LANG="en"


def mkcol(w):
    hx = hex(int(255*w))[2:]
    return '#000000' + hx

def _dep(labels, links):
    out = r"""\
<pre>
\begin{{dependency}}[hide label]
\begin{{deptext}}
{}\\
\end{{deptext}}
{}
\end{{dependency}}
</pre>
"""
    deptext = " \& ".join(labels)
    depedges = []
    for h, m, w in links:
        style = "[show label]" if w < 1 else ""
        depedges.append(f"\\depedge{style}{{{h+1}}}{{{m+1}}}{{{w}}}")
    return out.format(deptext, "\n".join(depedges))


def _svg(labels, links):
    head = """\
    digraph G{
    edge [dir=forward]
    node [shape=plaintext]"""

    labels = [w.replace('"', r'\"') for w in labels]

    lbl_code = ['{} [label="{}"]'.format(k, w)
                for k, w in enumerate(labels)]

    link_code = [f'{m} -> {h} [color="{mkcol(w)}"]' for h, m, w in links]

    dot_string = "\n".join([head] + lbl_code + link_code + ["}"])

    try:
        process = subprocess.Popen(
            ['dot', '-Tsvg'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
    except OSError:
        raise Exception('Cannot find the dot binary from Graphviz package')
    out, err = process.communicate(dot_string)
    if err:
        raise Exception(
            'Cannot create svg representation by running dot from string: {}'
            ''.format(dot_string))

    return out


if __name__ == '__main__':

    fpred = f"{LANG}_valid_predictions.txt"
    ftrue = f"data/{LANG}-valid.txt"

    toks = []
    parse_true = []
    arcs_pred = []
    parses_pred = []

    with open(ftrue) as f:
        for line in f:
            sent, _, _, true_tree = line.strip().split("\t")
            toks.append(sent.split())
            parse_true.append(true_tree)

    with open(fpred) as f:
        for line in f:
            if line.startswith("[dynet"):
                continue
            arcs, *trees = line.strip().split("\t")
            arcs_pred.append(arcs)
            parses_pred.append(trees)

    print(len(arcs_pred), len(toks))

    pack = []
    for k, (sent, arcs, parses, hd) in enumerate(zip(toks, arcs_pred, parses_pred, parse_true)):

        # number of non-hard arcs
        arcs = [float(x) for x in arcs.split()]
        if 3 <= sum(1 for a in arcs if 0 < a < 1) <= 4:
            pack.append((sent, arcs, parses, hd, k))

    pack.sort(key=lambda x: len(x[0]))

    with open("out2.html", "w") as f:
        #  for sent, arcs, parses, hd in zip(toks, arcs_pred, parses_pred, parse_true):
        for sent, arcs, parses, hd, k in pack:

            #  if len(parses) != 2:
                #  continue

            print("<h4>", k, "</h4>", file=f)
            print("<p>", " ".join(sent), "</p>", file=f)
            print("<p>", hd, "</p>", file=f)
            for prs in parses:
                corr = prs[prs.find(' ') + 1:].strip() == hd.strip()
                print("<p>", prs, "GOOD" if corr else "", "</p>", file=f)


            k = 0
            arc_list = []
            for m in range(1, len(sent) + 1):
                for h in range(len(sent) + 1):
                    if arcs[k] > 0:
                        arc_list.append((h, m, arcs[k]))
                    k += 1

            svg = _svg(["*"] + sent, arc_list)
            print(svg, file=f)
            dep = _dep(["*"] + sent, arc_list)
            print(dep, file=f)
            print("<hr />", file=f)

