from rrt_base import RRT
import numpy as np


class RRTStar(RRT):
    class Node(RRT.Node):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.cost = 0.0

    def __init__(self, connect_circle_dist=50.0, **kwargs):
        super().__init__(**kwargs)
        self.connect_circle_dist = connect_circle_dist
        self.path, self.min_cost = self.__plan()

    def __plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # Create a random node inside the bounded environment
            rnd = self.get_random_node()
            # Find nearest node
            nearest_node = self.get_nearest_node(rnd)
            # Get new node by connecting rnd_node and nearest_node
            new_node = self.steer(nearest_node, rnd, self.max_extend_length)
            # If path between new_node and nearest node is not in collision:
            if not self.collision(new_node, nearest_node):
                near_inds = self.near_nodes_inds(new_node)
                # Connect the new node to the best parent in near_inds
                new_node = self.choose_parent(new_node, near_inds)
                self.node_list.append(new_node)
                # Rewire the nodes in the proximity of new_node if it improves their costs
                self.rewire(new_node, near_inds)
        last_index, min_cost = self.best_goal_node_index()
        if last_index:
            return self.final_path(self.goal), min_cost
        return None, min_cost

    def choose_parent(self, new_node, near_inds):
        """Set new_node.parent to the lowest resulting cost parent in near_inds and
        new_node.cost to the corresponding minimal cost
        """
        min_cost = np.inf
        best_near_node = None
        # modify here: Go through all near nodes and evaluate them as potential parent nodes by
        # 1) checking whether a connection would result in a collision,
        # 2) evaluating the cost of the new_node if it had that near node as a parent,
        # 3) picking the parent resulting in the lowest cost and updating
        #    the cost of the new_node to the minimum cost.
        near_nodes = self.near_nodes_inds(new_node)
        for near_node in [self.node_list[i] for i in near_nodes]:
            if not self.collision(near_node, new_node):
                new_cost = self.new_cost(near_node, new_node)
                if new_cost < min_cost:
                    best_near_node = near_node
                    min_cost = new_cost

        # Don't need to modify beyond here
        new_node.cost = min_cost
        new_node.parent = best_near_node
        return new_node

    def rewire(self, new_node, near_inds):
        """Rewire near nodes to new_node if this will result in a lower cost"""
        # modify here: Go through all near nodes and check whether rewiring them
        # to the new_node would:
        # A) Not cause a collision and
        # B) reduce their own cost.
        # If A and B are true, update the cost and parent properties of the node.
        for near_node in [self.node_list[i] for i in near_inds]:
            if not self.collision(
                near_node,
                new_node,
            ):
                new_cost = self.new_cost(new_node, near_node)
                if new_cost < near_node.cost:
                    near_node.cost = new_cost
                    near_node.parent = new_node

        # Don't need to modify beyond here
        self.propagate_cost_to_leaves(new_node)

    def best_goal_node_index(self):
        """Find the lowest cost node to the goal"""
        min_cost = np.inf
        best_goal_node_idx = None
        for i in range(len(self.node_list)):
            node = self.node_list[i]
            # Has to be in close proximity to the goal
            if self.dist_to_goal(node.p) <= self.max_extend_length:
                # Connection between node and goal needs to be collision free
                if not self.collision(self.goal, node):
                    # The final path length
                    cost = node.cost + self.dist_to_goal(node.p)
                    if node.cost + self.dist_to_goal(node.p) < min_cost:
                        # Found better goal node!
                        min_cost = cost
                        best_goal_node_idx = i
        return best_goal_node_idx, min_cost

    def near_nodes_inds(self, new_node):
        """Find the nodes in close proximity to new_node"""
        nnode = len(self.node_list) + 1
        r = self.connect_circle_dist * np.sqrt((np.log(nnode) / nnode))
        dlist = [np.sum(np.square((node.p - new_node.p))) for node in self.node_list]
        near_inds = [dlist.index(i) for i in dlist if i <= r**2]
        return near_inds

    def new_cost(self, from_node, to_node):
        """to_node's new cost if from_node were the parent"""
        return from_node.cost + self.dynamics.cost(from_node.p, to_node.p)

    def propagate_cost_to_leaves(self, parent_node):
        """Recursively update the cost of the nodes"""
        parents = set()
        parents.add(parent_node)
        while len(parents) > 0:
            new_parents = set()
            for node in self.node_list:
                for potential_parent in parents:
                    if node.parent == potential_parent:
                        node.cost = self.new_cost(potential_parent, node)
                        new_parents.add(node)
            parents = new_parents
