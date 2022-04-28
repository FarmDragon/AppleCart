import numpy as np
import matplotlib.pyplot as plt


class RRT:
    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.parent = None

    def __init__(
        self,
        start,
        goal,
        obstacle_list,
        bounds,
        max_extend_length=3.0,
        path_resolution=0.5,
        goal_sample_rate=0.05,
        max_iter=100,
        distance_function=None,
        extend_function=None,
        plt=plt,
    ):
        self.start = self.Node(start)
        self.goal = self.Node(goal)
        self.bounds = bounds
        self.max_extend_length = max_extend_length
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.obstacle_list = obstacle_list
        self.node_list = []
        self.distance_function = distance_function if distance_function else lambda a,b: np.linalg.norm(a - b)
        self.extend_function = extend_function if extend_function else lambda f, t: f+(t-f)*self.max_extend_length/self.distance_function(f,t)
        self.plt = plt
        self.path = self.__plan()

    def __plan(self):
        """Plans the path from start to goal while avoiding obstacles"""
        self.node_list = [self.start]
        for i in range(self.max_iter):
            # modify here:
            # 1) Create a random node (rnd_node) inside
            # the bounded environment
            # 2) Find nearest node (nearest_node)
            # 3) Get new node (new_node) by connecting
            # rnd_node and nearest_node. Hint: steer
            # 4) If the path between new_node and the
            # nearest node is not in collision, add it to the node_list
            rnd_node = self.get_random_node()
            nearest = self.get_nearest_node(self.node_list, rnd_node)
            new_node = self.steer(
                nearest, rnd_node, max_extend_length=self.max_extend_length
            )
            if not self.collision(new_node, nearest, self.obstacle_list):
                self.node_list.append(new_node)

            # Don't need to modify beyond here
            # If the new_node is very close to the goal, connect it
            # directly to the goal and return the final path
            if self.dist_to_goal(self.node_list[-1].p) <= self.max_extend_length:
                final_node = self.steer(
                    self.node_list[-1], self.goal, self.max_extend_length
                )
                if not self.collision(
                    final_node, self.node_list[-1], self.obstacle_list
                ):
                    return self.final_path(len(self.node_list) - 1)
        return None  # cannot find path

    def steer(self, from_node, to_node, max_extend_length=np.inf):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length.
        """
        new_node = self.Node(to_node.p)
        new_node.p = self.extend_function(from_node.p, to_node.p)
        new_node.parent = from_node
        return new_node

    def dist_to_goal(self, p):
        """Distance from p to goal"""
        return self.distance_function(p, self.goal.p)

    def get_random_node(self):
        """Sample random node inside bounds or sample goal point"""
        if np.random.rand() > self.goal_sample_rate:
            # Sample random point inside boundaries
            rnd = self.Node(
                np.random.rand(2) * (self.bounds[1] - self.bounds[0]) + self.bounds[0]
            )
        else:
            # Select goal point
            rnd = self.Node(self.goal.p)
        return rnd
 
    def get_nearest_node(self, node_list, node):
        """Find the nearest node in node_list to node"""
        dlist = [self.distance_function(node.p, n.p) for n in node_list]
        minind = dlist.index(min(dlist))
        return node_list[minind]

    @staticmethod
    def collision(node1, node2, obstacle_list):
        """Check whether the path connecting node1 and node2
        is in collision with anyting from the obstacle_list
        """
        p1 = node2.p
        p2 = node1.p
        for o in obstacle_list:
            center_circle = o[0:2]
            radius = o[2]
            d12 = p2 - p1  # the directional vector from p1 to p2
            # defines the line v(t) := p1 + d12*t going through p1=v(0) and p2=v(1)
            d1c = center_circle - p1  # the directional vector from circle to p1
            # t is where the line v(t) and the circle are closest
            # Do not divide by zero if node1.p and node2.p are the same.
            # In that case this will still check for collisions with circles
            t = d12.dot(d1c) / (d12.dot(d12) + 1e-7)
            t = max(0, min(t, 1))  # Our line segment is bounded 0<=t<=1
            d = p1 + d12 * t  # The point where the line segment and circle are closest
            is_collide = np.sum(np.square(center_circle - d)) < radius**2
            if is_collide:
                return True  # is in collision
        return False  # is not in collision

    def final_path(self, goal_ind):
        """Compute the final path from the goal node to the start node"""
        path = [self.goal.p]
        node = self.node_list[goal_ind]
        # modify here: Generate the final path from the goal node to the start node.
        # We will check that path[0] == goal and path[-1] == start
        while node:
            path.append(node.p)
            node = node.parent
        return path

    def draw_graph(self):
        for node in self.node_list:
            if node.parent:
                self.plt.plot(
                    [node.p[0], node.parent.p[0]], [node.p[1], node.parent.p[1]], "-g"
                )
        if self.path:
            plt.plot([x for (x, y) in self.path], [y for (x, y) in self.path], "-r")

    def plot(self):
        plt = self.plt
        plt.figure()
        ax = plt.gca()
        for o in self.obstacle_list:
            circle = plt.Circle((o[0], o[1]), o[2], color="k")
            ax.add_artist(circle)
        plt.axis(
            [
                self.bounds[0] - 0.5,
                self.bounds[1] + 0.5,
                self.bounds[0] - 0.5,
                self.bounds[1] + 0.5,
            ]
        )
        plt.plot(self.start.p[0], self.start.p[1], "xr", markersize=10)
        plt.plot(self.goal.p[0], self.goal.p[1], "xb", markersize=10)
        plt.legend(("start", "goal"), loc="upper left")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
