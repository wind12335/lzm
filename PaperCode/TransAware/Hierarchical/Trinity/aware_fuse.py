import pickle
import copy
from collections import deque
import networkx as nx
import matplotlib.pyplot as plt

def _len_and_str(string):
    """Returns a tuple of string length and string."""
    return (len(string), string)


def _update_colocation_group(op_graph, colocation_group_map):
    """Updates colocation groups of operators in op_graph with new mapping."""
    # pick the shortest group name among groups as new group name
    group_dict = {group: min(group_set, key=_len_and_str)
                  for group, group_set in colocation_group_map.items()}

    # print merged groups
    reverse_mapping = {}
    for prev_name, new_name in group_dict.items():
        if new_name in reverse_mapping:
            reverse_mapping[new_name].append(prev_name)
        else:
            reverse_mapping[new_name] = [prev_name]

    # update colocation group
    for _, op_data in op_graph.nodes.items():
        if isinstance(op_data['colocation_group'], list):
            new_group = None
            for colocation_group in op_data['colocation_group']:
                ret = group_dict.get(colocation_group, colocation_group)
                if new_group is None:
                    new_group = ret
                else:
                    assert new_group == ret, 'node=%s, cur=%s, new=%s' % (
                        op_data['name'], new_group, ret)
        else:
            prev_group_name = op_data['colocation_group']
            new_group = group_dict.get(prev_group_name, prev_group_name)

        op_data['colocation_group'] = new_group


class ColocationGroupMap():
    """Data structure that maps group to a set of groups that are co-located."""

    def __init__(self):
        self._map = {}

    def colocate(self, group1, group2):
        prev_group1 = self._map.get(group1, set([group1]))
        prev_group2 = self._map.get(group2, set([group2]))
        new_group = prev_group1 | prev_group2

        # update existing groups
        for group in new_group:
            self._map[group] = new_group

    def __getitem__(self, op):
        return self._map[op]

    def __len__(self):
        return len(self._map)

    def items(self):
        return self._map.items()


def _run_colocation_step(op_graph):
    """Check whether there are operators that can be co-located.

    When the output of an operator is consumed only by another operator,
    assign the same colocation group for them

    Returns:
      True if there are opeartors that can be co-located.
      False, otherwise.
    """
    colocation_candidates = ColocationGroupMap()

    for op_id, op_data in op_graph.nodes.items():
        # TODO: should we consider tensor-wise? not operator wise?
        out_edges = list(op_graph.out_edges(op_id))
        if len(out_edges) != 1:
            continue

        next_op_id = out_edges[0][1]

        next_op_data = op_graph.nodes[next_op_id]

        op_group = op_data['colocation_group']
        next_op_group = next_op_data['colocation_group']

        if op_group != next_op_group:
            # these two can be colocated
            colocation_candidates.colocate(op_group, next_op_group)

    if len(colocation_candidates) > 0:
        _update_colocation_group(op_graph, colocation_candidates)
        return True

    return False


def _run_one_step_merge(
        op_graph, fusion_check_disjoint_paths, ungrouped_mapping, op1, op2, allow_cycle=False):
    """Generates a fused op graph.

    This first identifies ops that can be fused.
    When ops are in the same colocation group and they are directly
    connected, we fuse two ops into a single op.
    """
    def _add_test(op1, op2, op_graph_):
        op_graph = copy.deepcopy(op_graph_)
        pre = set(op_graph.predecessors(op1)) | set(op_graph.predecessors(op2))
        if op1 in pre:
            pre.remove(op1)
        if op2 in pre:
            pre.remove(op2)
        suc = set(op_graph.successors(op1)) | set(op_graph.successors(op2))
        if op1 in suc:
            suc.remove(op1)
        if op2 in suc:
            suc.remove(op2)
        op_graph.add_node("test")
        op_graph.add_edges_from([(p, "test") for p in pre])
        op_graph.add_edges_from([("test", s) for s in suc])
        op_graph.remove_nodes_from([op1, op2])
        flag = nx.is_directed_acyclic_graph(op_graph)
        op_graph.remove_nodes_from(["test"])
        return flag

    def _add_fused_edge(fused_op_graph, from_op_id, to_op_id, edge_data):
        """Adds an edge to the fused op graph.
        Returns:
            True if a new edge is added. False, otherwise.
        """
        if fused_op_graph.has_edge(from_op_id, to_op_id):
            # update existing edge
            prev_edge = fused_op_graph[from_op_id][to_op_id]
            prev_edge_tensors = prev_edge['tensor']
            tensors_to_add = [
                tensor_data_ for tensor_data_ in edge_data['tensor']
                if tensor_data_ not in prev_edge_tensors]
            for tensor_data_ in tensors_to_add:
                # prev_edge['weight'] += tensor_data_['weight']
                prev_edge['tensor'].append(tensor_data_)
            return False

        fused_op_graph.add_edge(from_op_id, to_op_id, **edge_data)
        return True

    # fused_op_graph = copy.deepcopy(op_graph)

    # internal edge might be connected to a fused op
    # todo 什么意思，为什么都成立？？
    if not op_graph.has_edge(op1, op2):
        print("op1 and op2 can not merge!")
        return False

    # check whether there is another path from op1 to op2
    if not allow_cycle and (op_graph.out_degree(op1) > 1
                            and op_graph.in_degree(op2) > 1):
        if fusion_check_disjoint_paths:
            # CAVEATS: finding disjoint paths may take long time
            paths = list(
                nx.node_disjoint_paths(op_graph, op1, op2))
            if len(paths) > 1:
                # ops cannot be fused since it will create a cycle
                print("ops cannot be fused since it will create a cycle")
                return False
        else:
            if not _add_test(op1, op2, op_graph):
                # skip this fusion due to potential cycle generation
                print("skip this fusion due to potential cycle generation")
                return False

    # fuse op2 into op1
    op1_data = op_graph.nodes[op1]
    op2_data = op_graph.nodes[op2]

    print("%s[%d] is fused into %s[%d]",
          op2_data["name"], op2, op1_data["name"], op1)

    op1_data["cost"] += op2_data["cost"]
    op1_data["mem"] += op2_data["mem"]
    # use max since each op runs at a time, not simultaneously
    # op1_data["temporary_memory"] = max(
    #     op1_data["temporary_memory"], op2_data["temporary_memory"])

    # add op2's edges to op1
    for in_edge in op_graph.in_edges(op2, data=True):
        from_op, _, edge_data = in_edge
        if from_op == op1:
            continue
        _add_fused_edge(
                op_graph, from_op, op1, edge_data)

    for out_edge in op_graph.out_edges(op2, data=True):
        _, to_op, edge_data = out_edge
        if to_op == op1:
            continue
        _add_fused_edge(
                op_graph, op1, to_op, edge_data)

    # op2 might be a fused op. merge information
    new_fused_ops = op1_data.get("aggregated_nodes", [])
    new_fused_ops += op2_data.pop("aggregated_nodes", [])
    new_fused_ops += [op2_data]
    op1_data["aggregated_nodes"] = new_fused_ops

    op_graph.remove_node(op2)

    def get_keys(d, value):
        return [k for k, v in d.items() if v == value]

    for key in get_keys(ungrouped_mapping, op2):
        ungrouped_mapping[key] = op1
    # update output memory
    # CAVEATS: output port number is no longer valid.
    output_tensors = {}
    for out_edge in op_graph.out_edges(op1, data=True):
        out_edge_data = out_edge[-1]
        for tensor_data in out_edge_data['tensor']:
            output_tensors[tensor_data['name']] = \
                tensor_data['num_bytes']
    op1_data['out_memory'] = list(output_tensors.values())
    op1_data['out_size'] = sum(list(output_tensors.values()))
    # need to assign new node ids and edge ids to be compatible with m_sct.
    # m_sct expects op_graph that has consecutive node ids and edge ids.
    func = lambda z: dict([(x, y) for y, x in z.items()])
    text = func(func(ungrouped_mapping)).values()
    print("节点归并进程：" + str(len(text)) + "\t" + str(len(op_graph.nodes)))
    return True


def _generate_fused_op_graph(
        op_graph, fusion_check_disjoint_paths, ungrouped_mapping, allow_cycle=False):
    """Generates a fused op graph.

    This first identifies ops that can be fused.
    When ops are in the same colocation group and they are directly
    connected, we fuse two ops into a single op.
    """
    def _assign_new_ids(fused_op_graph, op_name_old, op_name_new):
        """Returns a new graph that of which nodes have unique sequential ids.
        """
        new_fused_op_graph = nx.DiGraph()
        fused_id_map = {}  # maps ids in op_graph to new ids in fused_op_graph
        num_fused_ops = 0

        for new_id, (old_id, data) in enumerate(fused_op_graph.nodes.items()):
            # update id information
            data["old_name"] = old_id
            if old_id == op_name_old:
                new_id = op_name_new
            data["name"] = new_id
            new_fused_op_graph.add_node(new_id, **data)
            fused_id_map[old_id] = new_id

            # log fused op information
            if "aggregated_nodes" in data:
                num_fused_ops += len(data["aggregated_nodes"])
                fused_ops_list = [
                    "%s" % (fused_op_data["name"])
                    for fused_op_data in data["aggregated_nodes"]]

        for new_id, (u, v, data) in enumerate(fused_op_graph.edges(data=True)):
            data["old_name"] = data["name"]
            data["name"] = new_id
            new_fused_op_graph.add_edge(
                fused_id_map[u], fused_id_map[v], **data)

        return new_fused_op_graph

    def _add_fused_edge(fused_op_graph, from_op_id, to_op_id, edge_data):
        """Adds an edge to the fused op graph.
        Returns:
            True if a new edge is added. False, otherwise.
        """
        if fused_op_graph.has_edge(from_op_id, to_op_id):
            # update existing edge
            prev_edge = fused_op_graph[from_op_id][to_op_id]
            prev_edge_tensors = prev_edge['tensor']
            tensors_to_add = [
                tensor_data_ for tensor_data_ in edge_data['tensor']
                if tensor_data_ not in prev_edge_tensors]
            for tensor_data_ in tensors_to_add:
                # prev_edge['weight'] += tensor_data_['weight']
                prev_edge['tensor'].append(tensor_data_)
            return False

        fused_op_graph.add_edge(from_op_id, to_op_id, **edge_data)
        return True

    # pylint: disable=too-many-locals,too-many-branches
    def _create_colocation_group_to_ops_map(op_graph):
        """Generate a dict that maps a colocation group to its op id list."""
        retval = {}

        for op_id, op_data in op_graph.nodes().items():
            # assume there is only one group
            group = op_data['colocation_group']
            if group in retval:
                retval[group].append(op_id)
            else:
                retval[group] = [op_id]

        return retval

    print("Allow cycle in operator fusion: %s", str(allow_cycle))

    fused_op_graph = copy.deepcopy(op_graph)

    group_to_ops = _create_colocation_group_to_ops_map(
        fused_op_graph)

    for ops in group_to_ops.values():
        internal_edges = deque(
            [(u, v) for u, v in fused_op_graph.edges(ops)
             if u in ops and v in ops])

        while len(internal_edges) > 0:
            op1, op2 = internal_edges.popleft()

            # internal edge might be connected to a fused op
            if not fused_op_graph.has_edge(op1, op2):
                continue

            # check whether there is another path from op1 to op2
            if not allow_cycle and (fused_op_graph.out_degree(op1) > 1
                                    and fused_op_graph.in_degree(op2) > 1):
                if fusion_check_disjoint_paths:
                    # CAVEATS: finding disjoint paths may take long time
                    paths = list(
                        nx.node_disjoint_paths(fused_op_graph, op1, op2))
                    if len(paths) > 1:
                        # ops cannot be fused since it will create a cycle
                        continue
                else:
                    # if _add_test(op1, op2, op_graph):
                        # skip this fusion due to potential cycle generation
                    continue

            # fuse op2 into op1
            op1_data = fused_op_graph.nodes[op1]
            op2_data = fused_op_graph.nodes[op2]

            print("%s[%d] is fused into %s[%d]",
                  op2_data["name"], op2, op1_data["name"], op1)

            op1_data["cost"] += op2_data["cost"]
            op1_data["mem"] += op2_data["mem"]
            # use max since each op runs at a time, not simultaneously
            # op1_data["temporary_memory"] = max(
            #     op1_data["temporary_memory"], op2_data["temporary_memory"])

            # add op2's edges to op1
            for in_edge in fused_op_graph.in_edges(op2, data=True):
                from_op, _, edge_data = in_edge
                if from_op == op1:
                    continue
                if _add_fused_edge(
                        fused_op_graph, from_op, op1, edge_data):
                    # if the new edge is a candidate,
                    # check it by adding it to internal_edges
                    if from_op in ops:
                        internal_edges.append((from_op, op1))

            for out_edge in fused_op_graph.out_edges(op2, data=True):
                _, to_op, edge_data = out_edge
                if to_op == op1:
                    continue
                if _add_fused_edge(
                        fused_op_graph, op1, to_op, edge_data):
                    # if the new edge is a candidate,
                    # check it by adding it to internal_edges
                    if to_op in ops:
                        internal_edges.append((op1, to_op))

            # op2 might be a fused op. merge information
            new_fused_ops = op1_data.get("aggregated_nodes", [])
            new_fused_ops += op2_data.pop("aggregated_nodes", [])
            new_fused_ops += [op2_data]
            op1_data["aggregated_nodes"] = new_fused_ops

            fused_op_graph.remove_node(op2)

            def get_keys(d, value):
                return [k for k, v in d.items() if v == value]

            for key in get_keys(ungrouped_mapping, op2):
                ungrouped_mapping[key] = op1
            ops.remove(op2)
            # update output memory
            # CAVEATS: output port number is no longer valid.
            output_tensors = {}
            for out_edge in fused_op_graph.out_edges(op1, data=True):
                out_edge_data = out_edge[-1]
                for tensor_data in out_edge_data['tensor']:
                    output_tensors[tensor_data['name']] = \
                        tensor_data['num_bytes']
            op1_data['out_memory'] = list(output_tensors.values())
            op1_data['out_size'] = sum(list(output_tensors.values()))

    # need to assign new node ids and edge ids to be compatible with m_sct.
    # m_sct expects op_graph that has consecutive node ids and edge ids.
    return fused_op_graph


def merge_nodes(op_graph, ungrouped_mapping):
    out_dict = {}
    for i, node in enumerate(nx.topological_sort(op_graph)):
        out_dict[node] = op_graph.nodes[node]["out_size"]
    while len(out_dict.keys()) > 0:
        if len(op_graph.nodes) <= 250:
            break
        node = min(out_dict, key=out_dict.get)
        succs = op_graph.successors(node)
        pres = op_graph.predecessors(node)
        min_node = ''
        min_value = 100000000
        for i, nei in enumerate(succs):
            if min_value > op_graph.nodes[nei]["out_size"]:
                op_graph.nodes[nei]["colocation_group"] = op_graph.nodes[node]["colocation_group"]
        for i, nei in enumerate(pres):
            if min_value > op_graph.nodes[nei]["out_size"]:
                op_graph.nodes[nei]["colocation_group"] = op_graph.nodes[node]["colocation_group"]
        out_dict.pop(node)
    return _generate_fused_op_graph(op_graph, False, ungrouped_mapping)


def merge_nodes_2(op_graph, ungrouped_mapping):
    op_graph_fu = op_graph
    out_dict = {}
    for i, node in enumerate(nx.topological_sort(op_graph_fu)):
        out_dict[node] = op_graph_fu.nodes[node]["out_size"]
    while len(out_dict.keys()) > 0:
        if len(op_graph_fu.nodes) <= 250:
            break
        node = min(out_dict, key=out_dict.get)
        succs = op_graph_fu.successors(node)
        pres = op_graph_fu.predecessors(node)
        min_node = ''
        flag = False
        min_value = 100000000
        for i, nei in enumerate(succs):
            if min_value > op_graph_fu.nodes[nei]["out_size"]:
                min_node = nei
                # op_graph = _run_one_step_merge(op_graph, True, ungrouped_mapping, node, nei, False)
                # op_graph.nodes[nei]["colocation_group"] = op_graph.nodes[node]["colocation_group"]
        for i, nei in enumerate(pres):
            if min_value > op_graph_fu.nodes[nei]["out_size"]:
                min_node = nei
                # op_graph = _run_one_step_merge(op_graph, True, ungrouped_mapping, nei, node, False)
                # op_graph.nodes[nei]["colocation_group"] = op_graph.nodes[node]["colocation_group"]
                flag = True
        if not min_node:
            if node in out_dict.keys():
                out_dict.pop(node)
        else:
            if flag:
                _run_one_step_merge(op_graph_fu, False, ungrouped_mapping, min_node, node, False)
                if node in out_dict.keys():
                    out_dict.pop(node)
            else:
                if _run_one_step_merge(op_graph_fu, False, ungrouped_mapping, node, min_node, False):
                    if min_node in out_dict.keys():
                        out_dict.pop(min_node)
                else:
                    if node in out_dict.keys():
                        out_dict.pop(node)
    return _generate_fused_op_graph(op_graph_fu, False, ungrouped_mapping)


def main():
    div_path = "llll.pkl"
    with open(div_path, "rb") as file:
        j = pickle.load(file)
        mg, op_perf, step_stats, G, ungrouped_mapping = j['optim_mg'], j['op_perf'], j['step_stats'], j['G'], j[
            'ungrouped_mapping']
        func = lambda z: dict([(x, y) for y, x in z.items()])
        fused_G = _generate_fused_op_graph(G, False, ungrouped_mapping, allow_cycle=False)
        size = 0
        while 300 <= len(fused_G.nodes) != size:
            size = len(fused_G.nodes)
            _run_colocation_step(fused_G)
            fused_G = _generate_fused_op_graph(fused_G, False, ungrouped_mapping, allow_cycle=False)
            text = func(func(ungrouped_mapping)).values()
            print("节点归并进程_1：" + str(len(text)) + "\t" + str(len(fused_G.nodes)))
        size = 0
        while 300 <= len(fused_G.nodes) != size:
            size = len(fused_G.nodes)
            _run_colocation_step(fused_G)
            fused_G = _generate_fused_op_graph(fused_G, False, ungrouped_mapping, allow_cycle=False)
            text = func(func(ungrouped_mapping)).values()
            print("节点归并进程_1：" + str(len(text)) + "\t" + str(len(fused_G.nodes)))
        size = 0
        while 300 <= len(fused_G.nodes) != size:
            size = len(fused_G.nodes)
            fused_G = merge_nodes(fused_G, ungrouped_mapping)
            text = func(func(ungrouped_mapping)).values()
            print("节点归并进程_2：" + str(len(text)) + "\t" + str(len(fused_G.nodes)))
        size = 0
        while 300 <= len(fused_G.nodes) != size:
            size = len(fused_G.nodes)
            fused_G = merge_nodes_2(fused_G, ungrouped_mapping)
            text = func(func(ungrouped_mapping)).values()
            print("节点归并进程_3：" + str(len(text)) + "\t" + str(len(fused_G.nodes)))
        pos = nx.kamada_kawai_layout(fused_G)
        nc = nx.draw_networkx(fused_G, pos,
                                  nodelist=fused_G.nodes,
                                  node_color="#f555ff",
                                  with_labels=False,
                                  node_size=10,
                                  font_size=3,
                                  width=.5,
                                  arrowsize=5,
                                  cmap='plasma')
        plt.show()
        with open("input_1.pkl", 'wb') as file:
            pickle.dump({"optim_mg": mg, "op_perf": op_perf, "step_stats": step_stats, "G": fused_G,
                         "ungrouped_mapping": ungrouped_mapping}, file)


if __name__ == "__main__":
    main()
