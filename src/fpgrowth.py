"""
Native FP-Growth Algorithm & Association Rule Miner
Author: ASBD Hackathon Team
Implementation of Han et al. (2000): Mining Frequent Patterns without Candidate Generation
Compatible with mlxtend DataFrame interfaces.
"""

from collections import defaultdict
from itertools import combinations
import pandas as pd
import numpy as np

class FPNode:
    def __init__(self, item, count, parent):
        self.item = item
        self.count = count
        self.parent = parent
        self.children = {}
        self.next_homonym = None # Header table pointer

class FPTree:
    def __init__(self, transactions, min_sup_count):
        self.min_sup_count = min_sup_count
        self.header_table = {} # item: [count, head_node]
        self.root = FPNode(None, 0, None)
        self._build(transactions)

    def _build(self, transactions):
        # 1. Count frequencies
        item_counts = defaultdict(int)
        for trans in transactions:
            for item in set(trans):
                item_counts[item] += 1

        # 2. Filter frequent items
        for item, count in item_counts.items():
            if count >= self.min_sup_count:
                self.header_table[item] = [count, None]

        # 3. Sort frequent items descending by count
        sorted_items = sorted(self.header_table.keys(), key=lambda x: (-self.header_table[x][0], str(x)))

        # 4. Insert each transaction into the tree
        for trans in transactions:
            # Filter and sort items according to global order
            filtered_trans = [item for item in sorted_items if item in trans]
            if filtered_trans:
                self._insert_tree(filtered_trans, self.root)

    def _insert_tree(self, items, current_node):
        first = items[0]
        if first in current_node.children:
            current_node.children[first].count += 1
        else:
            new_node = FPNode(first, 1, current_node)
            current_node.children[first] = new_node
            # Link into header table
            if self.header_table[first][1] is None:
                self.header_table[first][1] = new_node
            else:
                curr = self.header_table[first][1]
                while curr.next_homonym is not None:
                    curr = curr.next_homonym
                curr.next_homonym = new_node

        # Recurse for remainder
        if len(items) > 1:
            self._insert_tree(items[1:], current_node.children[first])

def mine_fp_tree(tree, prefix, frequent_itemsets, total_transactions):
    # Sort items in header table ascending by count
    sorted_items = sorted(tree.header_table.keys(), key=lambda x: (tree.header_table[x][0], str(x)))

    for item in sorted_items:
        count = tree.header_table[item][0]
        new_prefix = prefix | frozenset([item])
        frequent_itemsets.append((new_prefix, count, count / total_transactions))

        # Find prefix paths
        conditional_patterns = []
        node = tree.header_table[item][1]
        while node is not None:
            path = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                # Replicate path by count
                for _ in range(node.count):
                    conditional_patterns.append(path)
            node = node.next_homonym

        # Build conditional tree
        cond_tree = FPTree(conditional_patterns, tree.min_sup_count)
        if cond_tree.header_table:
            mine_fp_tree(cond_tree, new_prefix, frequent_itemsets, total_transactions)

def fpgrowth(df_encoded_or_trans, min_support=0.1, use_colnames=True):
    """
    Computes frequent itemsets using FP-Growth.
    Compatible with mlxtend format.
    """
    if isinstance(df_encoded_or_trans, pd.DataFrame):
        # Convert boolean DataFrame to transaction list
        transactions = []
        columns = df_encoded_or_trans.columns
        for row in df_encoded_or_trans.itertuples(index=False):
            transactions.append([col for col, val in zip(columns, row) if val])
    else:
        transactions = df_encoded_or_trans

    N = len(transactions)
    min_sup_count = int(np.ceil(min_support * N))
    if min_sup_count < 1:
        min_sup_count = 1

    tree = FPTree(transactions, min_sup_count)
    raw_fis = []
    mine_fp_tree(tree, frozenset(), raw_fis, N)

    df_res = pd.DataFrame([
        {'support': round(sup, 4), 'itemsets': itemset, 'count': cnt}
        for itemset, cnt, sup in raw_fis
    ])
    if not df_res.empty:
        df_res = df_res.sort_values(by='support', ascending=False).reset_index(drop=True)
    return df_res

def association_rules(df_frequent, metric="confidence", min_threshold=0.6):
    """
    Generates association rules from frequent itemsets.
    Compatible with mlxtend association_rules format.
    """
    itemset_to_support = dict(zip(df_frequent['itemsets'], df_frequent['support']))
    rules = []

    for _, row in df_frequent.iterrows():
        itemset = row['itemsets']
        sup_itemset = row['support']
        if len(itemset) < 2:
            continue

        # Generate non-empty proper subsets
        items = list(itemset)
        for r in range(1, len(items)):
            for ant_comb in combinations(items, r):
                ant = frozenset(ant_comb)
                con = itemset - ant
                sup_ant = itemset_to_support.get(ant)
                sup_con = itemset_to_support.get(con)

                if sup_ant and sup_ant > 0:
                    conf = sup_itemset / sup_ant
                    lift = conf / sup_con if sup_con and sup_con > 0 else 1.0
                    leverage = sup_itemset - (sup_ant * sup_con) if sup_con else 0.0
                    conviction = (1.0 - sup_con) / (1.0 - conf) if (1.0 - conf) > 1e-6 else np.inf

                    if metric == "confidence" and conf >= min_threshold:
                        rules.append({
                            'antecedents': ant,
                            'consequents': con,
                            'antecedent support': round(sup_ant, 4),
                            'consequent support': round(sup_con, 4) if sup_con else 0.0,
                            'support': round(sup_itemset, 4),
                            'confidence': round(conf, 4),
                            'lift': round(lift, 4),
                            'leverage': round(leverage, 4),
                            'conviction': round(conviction, 4) if conviction != np.inf else 999.0
                        })

    df_rules = pd.DataFrame(rules)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(by=['lift', 'confidence', 'support'], ascending=[False, False, False]).reset_index(drop=True)
    return df_rules
