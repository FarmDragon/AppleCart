from rrt_star import RRTStar
from rrt_base import RRT


class RGRRT(RRT):
    class Node(RRT.Node):
        def __init__(self, *args, reachable_states=set(), **kwargs):
            super().__init__(*args, **kwargs)
            self.reachable_states = reachable_states

    def get_nearest_node(self, node: "RGRRT.Node"):
        n: "RGRRT.Node" = super().get_nearest_node(node)
        return n
