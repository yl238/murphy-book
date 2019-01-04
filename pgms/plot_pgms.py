import daft

def plot_cpd():
    pgm = daft.PGM([4, 4], origin=[0.3, 0.3])

    pgm.add_node(daft.Node('G1', r"$G_1$", 1, 3))
    pgm.add_node(daft.Node('G2', r"$G_2$", 3, 3))

    pgm.add_node(daft.Node('X', r"$X$", 1, 1))
    pgm.add_node(daft.Node('Y', r"$Y$", 3, 1))
    pgm.add_node(daft.Node('T', r"$T$", 2, 2, observed=True))

    pdict = {'linewidth': 0.5, 'head_width':0.25, 'head_length':0.15}
    pgm.add_edge('G1', 'T', **pdict)
    pgm.add_edge('G2', 'T', **pdict)
    pgm.add_edge('T', 'X', **pdict)
    pgm.add_edge('T', 'Y', **pdict)
    pgm.render()
    pgm.figure.savefig('figures/simple_cpd.png',dpi=150)

if __name__=='__main__':
    plot_cpd()
