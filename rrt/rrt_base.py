from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


class RRT:
    class Dynamics:
        def __init__(self, max_extend_length) -> None:
            self.max_extend_length = max_extend_length

        def run_forward(self, f, t):
            return RRT.Node(f + (t - f) * self.max_extend_length / self.distance(f, t))

        def cost(self, f, t):
            return self.distance(t, f)

        def distance(self, f, t):
            return np.linalg.norm(t - f)

    class Node:
        def __init__(self, p):
            self.p = np.array(p)
            self.p.astype(np.float64, copy=False)
            self.parent = None
            self.u = None
            self.reachables = None

    def __init__(
        self,
        start,
        goal,
        bounds,
        obstacle_list=[],
        max_extend_length=0.5,
        path_resolution=0.5,
        goal_sample_rate=0.05,
        max_iter=100,
        dynamics=None,
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
        self.plt = plt
        self.dynamics = dynamics if dynamics else RRT.Dynamics(max_extend_length)
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
            nearest = self.get_nearest_node(rnd_node)
            new_node = self.steer(
                nearest, rnd_node, max_extend_length=self.max_extend_length
            )
            if not self.collision(new_node, nearest):
                self.node_list.append(new_node)

            # Don't need to modify beyond here
            # If the new_node is very close to the goal, connect it
            # directly to the goal and return the final path
            if self.dist_to_goal(self.node_list[-1].p) <= self.max_extend_length:
                final_node = self.steer(
                    self.node_list[-1], self.goal, self.max_extend_length
                )
                if not self.collision(final_node, self.node_list[-1]):
                    return self.final_path(self.goal)
        return None  # cannot find path

    def steer(self, from_node, to_node, max_extend_length=np.inf):
        """Connects from_node to a new_node in the direction of to_node
        with maximum distance max_extend_length.
        """
        new_node = self.dynamics.run_forward(from_node.p, to_node.p)
        new_node.parent = from_node
        return new_node

    def dist_to_goal(self, p):
        """Distance from p to goal"""
        return self.dynamics.distance(p, self.goal.p)

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

    def get_nearest_node(self, node):
        """Find the nearest node in node_list to node"""
        dlist = [self.dynamics.distance(node.p, n.p) for n in self.node_list]
        minind = dlist.index(min(dlist))
        return self.node_list[minind]

    def collision(self, node1, node2):
        """Check whether the path connecting node1 and node2
        is in collision with anyting from the obstacle_list
        """
        p1 = node2.p
        p2 = node1.p
        for o in self.obstacle_list:
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

    def final_path(self, goal):
        return [n.p for n in self.node_path(goal.p)]

    def node_path(self, goal_state):
        """Compute the final path from the goal node to the start node"""
        path = [self.goal]
        node = self.get_nearest_node(RRT.Node(goal_state))
        # modify here: Generate the final path from the goal node to the start node.
        # We will check that path[0] == goal and path[-1] == start
        while node:
            path.append(node)
            node = node.parent
        return path

    # Plotter urns a node into a 2d point to be plotted.
    def plot(self, plotter=None, bounds=None):
        plotter = plotter if plotter else lambda n: [n.p[0], n.p[1]]
        bounds = (
            bounds
            if bounds
            else [
                self.bounds[0] - 0.5,
                self.bounds[1] + 0.5,
                self.bounds[0] - 0.5,
                self.bounds[1] + 0.5,
            ]
        )
        plt = self.plt
        plt.figure()
        ax = plt.gca()
        for o in self.obstacle_list:
            circle = plt.Circle((o[0], o[1]), o[2], color="k")
            ax.add_artist(circle)
        plt.axis(
            [
                bounds[0],
                bounds[1],
                bounds[2],
                bounds[3],
            ]
        )
        start = plotter(self.start)
        goal = plotter(self.goal)
        plt.plot(start[0], start[1], "xr", markersize=10)
        plt.plot(goal[0], goal[1], "xb", markersize=10)
        plt.legend(("start", "goal"), loc="upper left")
        plt.gca().set_aspect("equal")
        plt.tight_layout()
        for node in self.node_list:
            if node.parent:
                self.plt.plot(
                    [node.p[0], node.parent.p[0]], [node.p[1], node.parent.p[1]], "-g"
                )
        if self.path:
            plt.plot([x for (x, y) in self.path], [y for (x, y) in self.path], "-r")

    def save(self, path=Path.cwd() / "rrt_path"):
        states = np.array([(n.p[0], n.p[1]) for n in self.node_path(self.goal.p)])
        np.save(
            path / "states.npy",
            states,
        )
        inputs = np.array([n.u for n in self.node_path(self.goal.p)])
        np.save(path / "inputs.npy", inputs)
        print(states)
        print(inputs)
