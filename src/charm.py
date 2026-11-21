"""
CHARM: Closed Association Rule Mining via IT-Tree Search
Author: ASBD Hackathon Team
Implementation of Zaki & Hsiao (2002): CHARM: An Efficient Algorithm for Closed Itemset Mining
"""

from typing import List, Set, Dict, Tuple
from collections import defaultdict

class CHARM:
    """
    CHARM algorithm for mining Closed Frequent Itemsets (CFIs) using Itemset-Tidset (IT) trees.
    Implements immutable candidate generation and IT-tree search based on Zaki & Hsiao (2002),
    followed by rigorous post-processing subsumption check to guarantee exact mathematical closure:
    an itemset X is closed iff no proper superset Y > X has support(Y) == support(X).
    """
    def __init__(self, min_support: float = 0.1):
        """
        :param min_support: minimum support threshold as a fraction in [0, 1]
        """
        self.min_support = min_support
        self.min_sup_count = 0
        self.num_transactions = 0
        self.closed_itemsets = []  # List of tuples: (frozenset(items), support_count)
        self.raw_tid_lists = defaultdict(set)

    def fit(self, transactions: List[List[str]]):
        self.num_transactions = len(transactions)
        self.min_sup_count = int(self.min_support * self.num_transactions)
        if self.min_sup_count < 1:
            self.min_sup_count = 1

        # 1. Build vertical tidset representation for each 1-item
        self.raw_tid_lists = defaultdict(set)
        for tid, trans in enumerate(transactions):
            for item in set(trans):
                self.raw_tid_lists[item].add(tid)

        # 2. Filter frequent 1-itemsets
        nodes = []
        for item, tids in self.raw_tid_lists.items():
            if len(tids) >= self.min_sup_count:
                nodes.append((frozenset([item]), tids))

        # Sort nodes by support ascending (standard CHARM heuristic)
        nodes.sort(key=lambda x: (len(x[1]), sorted(list(x[0]))))

        # 3. Recursive IT-tree exploration without in-place mutation
        candidates = []
        self._charm_extend(nodes, candidates)

        # 4. Rigorous post-processing subsumption check
        # An itemset X is closed iff no proper superset Y > X has support(Y) == support(X)
        by_sup = defaultdict(set)
        for itemset, tids in candidates:
            by_sup[len(tids)].add(itemset)

        self.closed_itemsets = []
        for sup, itemset_group in by_sup.items():
            for X in itemset_group:
                # Check if X is subsumed by any proper superset Y with identical support
                is_subsumed = any(len(Y) > len(X) and X.issubset(Y) for Y in itemset_group)
                if not is_subsumed:
                    self.closed_itemsets.append((X, sup))

        # Sort closed itemsets by support descending, then length, then items
        self.closed_itemsets.sort(key=lambda x: (-x[1], -len(x[0]), sorted(list(x[0]))))
        return self

    def _charm_extend(self, nodes: List[Tuple[frozenset, Set[int]]], candidates: List[Tuple[frozenset, Set[int]]]):
        """
        Recursive IT-tree extension procedure using immutable candidate lists.
        Applies CHARM pruning properties:
        - If Ti == Tj or Ti subset Tj: Xj appears in every transaction of Ti, so Xj is merged into curr_Xi.
        - Offspring nodes in new_nodes inherit curr_Xi closures.
        """
        for i in range(len(nodes)):
            Xi, Ti = nodes[i]
            curr_Xi = Xi
            new_nodes = []

            for j in range(i + 1, len(nodes)):
                Xj, Tj = nodes[j]
                Tij = Ti & Tj
                sup_ij = len(Tij)

                if sup_ij >= self.min_sup_count:
                    # Property 1 & Property 2: Ti == Tj or Ti is strict subset of Tj
                    # Xj appears in all transactions where Ti appears
                    if Ti == Tj or Ti.issubset(Tj):
                        curr_Xi = curr_Xi | Xj
                    else:
                        # Property 3 & Property 4: Tj is strict subset or general overlap
                        new_nodes.append((Xi | Xj, Tij))

            # All offspring nodes in new_nodes must inherit all items merged into curr_Xi
            updated_new_nodes = []
            for Xk, Tk in new_nodes:
                updated_new_nodes.append((Xk | curr_Xi, Tk))

            if updated_new_nodes:
                updated_new_nodes.sort(key=lambda x: (len(x[1]), sorted(list(x[0]))))
                self._charm_extend(updated_new_nodes, candidates)

            candidates.append((curr_Xi, Ti))

    def get_closed_itemsets(self):
        """Returns list of dicts: {'itemset': tuple, 'length': int, 'support_count': int, 'support': float}"""
        return [
            {
                'itemset': tuple(sorted(list(x[0]))),
                'length': len(x[0]),
                'support_count': x[1],
                'support': round(x[1] / self.num_transactions, 4)
            }
            for x in self.closed_itemsets
        ]
