#!/usr/bin/env python

# Load required modules
import sys, os, argparse, numpy as np, random
import scipy.io as io
################################################################################
# PROJECT 1A: MUTATION SIGNATURES
################################################################################
def mutation_signatures_sim_data(N, L, K, output_prefix, mloc=100, mscale=50):
    # Create the signatures
    p_alphas = [2./L] * L
    e_alphas = [1.] * K
    P = np.random.dirichlet(p_alphas, size=K)
    E = np.random.dirichlet(e_alphas, size=N)

    # Sample data
    samples = [ 'Sample-%s' % (i+1) for i in range(N) ]
    categories = [ 'C%s' % (i+1) for i in range(L) ]
    M = np.zeros((N, L), dtype=np.int)
    for i, (sample, e_i) in enumerate(zip(samples, E)):
        # Sample the number of mutations for this sample, and the exposures
        from scipy.stats import expon
        n_muts = int(expon.rvs(loc=mloc, scale=mscale, size=1)[0])
        e_i = np.random.dirichlet(e_alphas)

        # Generate mutations
        for _ in range(n_muts):
            k = np.where(np.random.multinomial(1, pvals=e_i) == 1)[0][0]
            j = np.where(np.random.multinomial(1, pvals=P[k]) == 1)[0][0]
            M[i, j] += 1

    # Add some Poisson noise
    M += np.random.poisson(1, size=(N, L))

    # Output to file
    np.save(output_prefix + '-signatures.npy', P)
    np.save(output_prefix + '-exposure.npy', E)
    np.save(output_prefix + '-mutation-counts.npy', M)
    with open(output_prefix + '-mutation-counts.tsv', 'w') as OUT:
        OUT.write('\t%s\n' % '\t'.join(categories))
        for i, sample in enumerate(samples):
            OUT.write('%s\t%s\n' % (sample, '\t'.join(map(str, M[i]))))

################################################################################
# PROJECT 1B: MUTUALLY EXCLUSIVE MUTATIONS
################################################################################
def mutually_exclusive_mutations_sim_data(N, M, bmr, output_prefix):
    # Generate samples
    assert(N >= 25)
    samples = [ 'Sample-%s' % (i+1) for i in range(N)]

    # Generate genes (we need at least 4)
    assert( M >= 4 )
    genes = [ 'Gene-%s' % (i+1) for i in range(M) ]

    # Generate mutation data, first implanting our known patterns
    from collections import defaultdict
    sampleToMutations = defaultdict(set)
    mut_samples = np.random.permutation(samples)[:N/2]
    for i in range(4):
        for s in mut_samples[i::4]:
            sampleToMutations[s].add( genes[i] )

    # Then adding noise (add default one mutation per sample)
    for s in samples: sampleToMutations[s].add( random.choice(genes) )
    for g in genes:
        for s in samples:
            if np.random.rand() < bmr:
                sampleToMutations[s].add( g )

    # Output to file each of the datasets
    with open(output_prefix + '-mutation-matrix.tsv', 'w') as OUT:
        OUT.write('\n'.join('%s\t%s' % (s, '\t'.join(sorted(sampleToMutations[s]))) for s in samples))

################################################################################
# PROJECT 1C: GENETIC INTERACTIONS
################################################################################
# Union-find data structure.  The implementation below is copied from the NetworkX implementation
# (see https://github.com/networkx/) but adds explicit root tracking and external access to the
# roots and to set weights.
#    Copyright (C) 2004-2011 by
#    Aric Hagberg <hagberg@lanl.gov>
#    Dan Schult <dschult@colgate.edu>
#    Pieter Swart <swart@lanl.gov>
#    All rights reserved.
#    BSD license.
class UnionFind:
    """Union-find data structure.

    Each unionFind instance X maintains a family of disjoint sets of
    hashable objects, supporting the following two methods:

    - X[item] returns a name for the set containing the given item.
      Each set is named by an arbitrarily-chosen one of its members; as
      long as the set remains unchanged it will keep the same name. If
      the item is not yet part of a set in X, a new singleton set is
      created for it.

    - X.union(item1, item2, ...) merges the sets containing each item
      into a single larger set.  If any item is not yet part of a set
      in X, it is added to X as one of the members of the merged set.

      Union-find data structure. Based on Josiah Carlson's code,
      http://aspn.activestate.com/ASPN/Cookbook/Python/Recipe/215912
      with significant additional changes by D. Eppstein.
      http://www.ics.uci.edu/~eppstein/PADS/UnionFind.py

    """

    def __init__(self, data=[]):
        """Create a new empty union-find structure."""
        self.weights = {}
        self.parents = {}
        self.roots = set()

        for datum in data:
            self.weights[datum] = 1
            self.parents[datum] = datum
            self.roots.add(datum)

    def __getitem__(self, obj):
        """Find and return the name of the set containing the obj."""

        # check for previously unknown obj
        if obj not in self.parents:
            self.parents[obj] = obj
            self.weights[obj] = 1
            self.roots.add(obj)
            return obj

        # find path of objects leading to the root
        path = [obj]
        root = self.parents[obj]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        """Iterate through all items ever found or unioned by this structure."""
        return iter(self.parents)

    def union(self, *objects):
        """Find the sets containing the objects and merge them all."""
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r],r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest
                self.roots.remove(r)

    def groups(self):
        groupIndex = 0
        groupToIndex = {}
        currentGroups = [[]]
        for n in self.parents:
            group = self[n]
            if group not in groupToIndex:
                groupToIndex[group] = groupIndex
                groupIndex += 1
            if len(currentGroups) <= groupToIndex[group]:
                currentGroups.append([]);
            currentGroups[groupToIndex[group]].append( +n )

        return currentGroups

def genetic_interactions_sim_data(N, output_prefix, m=2, random_seed=None, t=0.05):
    # Generate a random graph
    import networkx as nx
    G = nx.barabasi_albert_graph(N, m, seed=random_seed)
    nodes = sorted(G.nodes())
    edges = G.edges()

    # Assign each pair of nodes a score
    S = np.random.normal(loc=0.0, scale=1.0, size=(N, N))
    S = (S+S.T)/2
    for i, j in edges:
        S[i, j] = S[j, i] = -abs(np.random.normal(loc=-2.5, scale=2))

    # Run a heat diffusion
    import scipy as sp, scipy.linalg
    A = nx.to_numpy_matrix(G, dtype=np.float64, nodelist=nodes)
    D, V = sp.linalg.eigh(t*(A-np.diag(np.sum(A, axis=1))))
    F = np.dot(np.exp(D)*V, sp.linalg.inv(V))

    # Perform average-linkage clustering on the diffusion matrix to create
    # a hierarchy
    from scipy.cluster.hierarchy import linkage
    Z = linkage(F, method='average', metric='cosine')
    U = UnionFind()
    gene_sets = set()
    node_set = set(nodes)
    for i, (a, b, c, d) in enumerate(Z):
        U.union(int(a), int(b), N+i)
        gene_sets |= set( frozenset(set(g) & node_set) for g in U.groups() )

    gene_sets = sorted(gene_sets, key=lambda gs: len(gs))

    # Output to file
    np.save(output_prefix + '-genetic-interactions.npy', S)
    with open(output_prefix + '-gene-names.txt', 'w') as OUT:
        OUT.write('\n'.join(['Gene-%s' % (n+1) for n in nodes]))
    with open(output_prefix + '-hierarchy-sets.tsv', 'w') as OUT:
        OUT.write('\n'.join('\t'.join(['Gene-%s' % (g+1) for g in sorted(gs)]) for gs in gene_sets))

################################################################################
# MAIN
################################################################################
# Parse arguments
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_prefix', type=str, required=True)
    parser.add_argument('-rs', '--random_seed', type=int, required=False, default=87419287)
    subparser = parser.add_subparsers(dest='topic', help='Project 1 topic')

    # Project 1a
    a_parser = subparser.add_parser("a")
    a_parser.add_argument('-ns', '--n_samples', dest='N', type=int, required=True, help='We need at least 100.')
    a_parser.add_argument('-nc', '--n_cats', dest='L', type=int, required=True, help='We need at least 20.')
    a_parser.add_argument('-nt', '--n_topics', dest='K', type=int, required=True, help='We need at least 20.')

    # Project 1b
    b_parser = subparser.add_parser("b")
    b_parser.add_argument('-ns', '--n_samples', dest='N', type=int, required=True, help='We need at least 25.')
    b_parser.add_argument('-ng', '--n_genes', dest='M', type=int, required=True, help='We need at least four.')
    b_parser.add_argument('-b', '--bmr', type=float, required=True)

    # Project 1c
    c_parser = subparser.add_parser("c")
    c_parser.add_argument('-ng', '--n_genes', dest='N', type=int, required=True)

    return parser

def run( args ):
    # Set the random seed
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    # Generate simulated data
    if args.topic == 'a':
        mutation_signatures_sim_data(args.N, args.L, args.K, args.output_prefix)
    elif args.topic == 'b':
        mutually_exclusive_mutations_sim_data(args.N, args.M, args.bmr, args.output_prefix)
    elif args.topic == 'c':
        genetic_interactions_sim_data(args.N, args.output_prefix, random_seed=args.random_seed)
    else:
        raise NotImplementedError('Topic "%s" not implemented.' % args.topic)

if __name__ == '__main__': run( get_parser().parse_args(sys.argv[1:]) )
