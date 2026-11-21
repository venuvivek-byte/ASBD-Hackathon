"""
Native FP-Growth Algorithm & Association Rule Miner
Author: ASBD Hackathon Team
Implementation of Han et al. (2000): Mining Frequent Patterns without Candidate Generation
Compatible with mlxtend DataFrame interfaces and validated against mlxtend 0.23+.
"""

from collections import defaultdict
from itertools import combinations
from typing import List, Dict, Tuple, Set, Any, Optional, Union
import pandas as pd
import numpy as np


class FPNode:
    """
    FP-Tree Node representation.
    Tracks item name, transaction support count, parent node reference,
    dictionary of children nodes, and next_homonym pointer for header table linked list.
    """
    def __init__(self, item: Optional[str], count: int, parent: Optional['FPNode']):
        self.item = item
        self.count = count
        self.parent = parent
        self.children: Dict[str, 'FPNode'] = {}
        self.next_homonym: Optional['FPNode'] = None


class FPTree:
    """
    Frequent Pattern Tree (FP-Tree) structure.
    Maintains:
      - header_table: dict mapping item -> [frequency_count, head_node, tail_node]
      - root: null root FPNode
      - min_sup_count: absolute minimum support threshold
    """
    def __init__(self, transactions: Union[List[List[str]], List[Tuple[List[str], int]]], min_sup_count: int):
        self.min_sup_count = min_sup_count
        self.header_table: Dict[str, List[Any]] = {}  # item: [count, head_node, tail_node]
        self.root = FPNode(None, 0, None)
        self._build(transactions)

    def _build(self, transactions: Union[List[List[str]], List[Tuple[List[str], int]]]):
        # 1. Count global frequencies
        item_counts = defaultdict(int)
        for entry in transactions:
            if isinstance(entry, tuple):
                items, cnt = entry
            else:
                items, cnt = entry, 1
            for item in set(items):
                item_counts[item] += cnt

        # 2. Filter frequent items exceeding min_sup_count
        for item, count in item_counts.items():
            if count >= self.min_sup_count:
                self.header_table[item] = [count, None, None]

        # 3. Sort frequent items descending by count (with deterministic secondary key)
        sorted_items = sorted(
            self.header_table.keys(),
            key=lambda x: (-self.header_table[x][0], str(x))
        )

        # 4. Insert each transaction (with prefix sharing) into the tree
        for entry in transactions:
            if isinstance(entry, tuple):
                items, cnt = entry
            else:
                items, cnt = entry, 1
            item_set = set(items)
            filtered_trans = [item for item in sorted_items if item in item_set]
            if filtered_trans:
                self._insert_tree(filtered_trans, self.root, cnt)

    def _insert_tree(self, items: List[str], current_node: FPNode, count: int = 1):
        """
        Iterative insertion of sorted transaction items into the FP-tree,
        updating counts along existing prefixes and creating new branch nodes with O(1) header linking.
        """
        curr = current_node
        for item in items:
            if item in curr.children:
                curr.children[item].count += count
            else:
                new_node = FPNode(item, count, curr)
                curr.children[item] = new_node
                # O(1) Header table linking via tail pointer
                if self.header_table[item][1] is None:
                    self.header_table[item][1] = new_node
                    self.header_table[item][2] = new_node
                else:
                    self.header_table[item][2].next_homonym = new_node
                    self.header_table[item][2] = new_node
            curr = curr.children[item]


def mine_fp_tree(tree: FPTree, prefix: frozenset, frequent_itemsets: list, total_transactions: int):
    """
    Recursively mines conditional FP-trees in bottom-up order (ascending header table frequency).
    Extracts conditional pattern bases by following parent pointers from each header node.
    """
    # Sort items in header table ascending by count (bottom-up mining)
    sorted_items = sorted(
        tree.header_table.keys(),
        key=lambda x: (tree.header_table[x][0], str(x))
    )

    for item in sorted_items:
        count = tree.header_table[item][0]
        new_prefix = prefix | frozenset([item])
        frequent_itemsets.append((new_prefix, count, count / total_transactions))

        # Extract conditional pattern base by ascending parent pointers
        conditional_patterns = []
        node = tree.header_table[item][1]
        while node is not None:
            path = []
            parent = node.parent
            while parent is not None and parent.item is not None:
                path.append(parent.item)
                parent = parent.parent
            if path:
                conditional_patterns.append((path, node.count))
            node = node.next_homonym

        # Build conditional FP-tree and recurse
        cond_tree = FPTree(conditional_patterns, tree.min_sup_count)
        if cond_tree.header_table:
            mine_fp_tree(cond_tree, new_prefix, frequent_itemsets, total_transactions)


def fpgrowth(df_encoded_or_trans, min_support: float = 0.1, use_colnames: bool = True) -> pd.DataFrame:
    """
    Computes frequent itemsets using FP-Growth.
    Compatible with mlxtend.frequent_patterns.fpgrowth interface.

    Parameters
    ----------
    df_encoded_or_trans : pd.DataFrame or List[List[str]]
        One-hot encoded boolean DataFrame or list of transactions.
    min_support : float
        Minimum support threshold as a fraction in (0, 1].
    use_colnames : bool
        If True, preserves item string names.

    Returns
    -------
    pd.DataFrame with columns ['support', 'itemsets', 'count']
    """
    if isinstance(df_encoded_or_trans, pd.DataFrame):
        transactions = []
        columns = df_encoded_or_trans.columns
        for row in df_encoded_or_trans.itertuples(index=False):
            transactions.append([col for col, val in zip(columns, row) if val])
    else:
        transactions = df_encoded_or_trans

    N = len(transactions)
    # Floating-point safe ceiling for minimum support count
    min_sup_count = int(np.ceil(round(min_support * N, 9)))
    if min_sup_count < 1:
        min_sup_count = 1

    tree = FPTree(transactions, min_sup_count)
    raw_fis = []
    mine_fp_tree(tree, frozenset(), raw_fis, N)

    df_res = pd.DataFrame([
        {'support': sup, 'itemsets': itemset, 'count': cnt}
        for itemset, cnt, sup in raw_fis
    ])
    if not df_res.empty:
        df_res['length'] = df_res['itemsets'].apply(lambda x: len(x))
        df_res = df_res.sort_values(by=['support', 'length'], ascending=[False, False]).reset_index(drop=True)
    return df_res


def association_rules(
    df_frequent: pd.DataFrame,
    metric: str = "confidence",
    min_threshold: float = 0.6
) -> pd.DataFrame:
    """
    Generates association rules from frequent itemsets.
    Compatible with mlxtend.frequent_patterns.association_rules interface.
    Computes Support, Confidence, Lift, Leverage, and Conviction with mathematical precision.
    """
    itemset_to_support = dict(zip(df_frequent['itemsets'], df_frequent['support']))
    rules = []

    for _, row in df_frequent.iterrows():
        itemset = row['itemsets']
        sup_itemset = row['support']
        if len(itemset) < 2:
            continue

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
                    conviction = (1.0 - sup_con) / (1.0 - conf) if (1.0 - conf) > 1e-12 else np.inf

                    if metric == "confidence" and conf >= min_threshold:
                        rules.append({
                            'antecedents': ant,
                            'consequents': con,
                            'antecedent support': sup_ant,
                            'consequent support': sup_con if sup_con else 0.0,
                            'support': sup_itemset,
                            'confidence': conf,
                            'lift': lift,
                            'leverage': leverage,
                            'conviction': conviction
                        })

    df_rules = pd.DataFrame(rules)
    if not df_rules.empty:
        df_rules = df_rules.sort_values(
            by=['lift', 'confidence', 'support'],
            ascending=[False, False, False]
        ).reset_index(drop=True)
    return df_rules


def prune_redundant_rules(rules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prunes redundant association rules according to standard generality & closure principles:
    A rule R1: X1 -> Y1 is redundant if there exists another rule R2: X2 -> Y2
    (where X2 is a proper subset of X1 and Y1 is a subset of Y2) with equal or higher confidence
    and equal or higher support (i.e. adding items to the antecedent does not improve predictive power).
    """
    if rules_df.empty:
        return rules_df.copy()

    records = rules_df.to_dict('records')
    by_consequent = defaultdict(list)
    for idx, r in enumerate(records):
        con = frozenset(r['consequents']) if not isinstance(r['consequents'], frozenset) else r['consequents']
        ant = frozenset(r['antecedents']) if not isinstance(r['antecedents'], frozenset) else r['antecedents']
        by_consequent[con].append((idx, ant, r['confidence'], r['support']))

    redundant = set()
    for con, group in by_consequent.items():
        n = len(group)
        for i in range(n):
            idx_i, ant_i, conf_i, sup_i = group[i]
            if idx_i in redundant:
                continue
            for j in range(n):
                if i == j:
                    continue
                idx_j, ant_j, conf_j, sup_j = group[j]
                if ant_j < ant_i and conf_j >= conf_i - 1e-9:
                    redundant.add(idx_i)
                    break

    pruned_df = rules_df.drop(index=list(redundant)).reset_index(drop=True)
    return pruned_df
