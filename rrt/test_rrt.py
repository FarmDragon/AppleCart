from rrt import (RRT, RRTStar)
import numpy as np
import matplotlib.pyplot as plt

start = np.array([11, 0]) # Start location
goal = np.array([6, 8]) # Goal location

obstacles = [ # circles parametrized by [x, y, radius]
        np.array([9, 6, 2]),
        np.array([9, 8, 1]),
        np.array([9, 10, 2]),
        np.array([4, 5, 2]),
        np.array([7, 5, 2]),
        np.array([4, 10, 1])
] 

bounds = np.array([-2, 15]) # Bounds in both x and y

def plot_scene(obstacle_list, start, goal):
    ax = plt.gca()
    for o in obstacle_list:
        circle = plt.Circle((o[0], o[1]), o[2], color='k')
        ax.add_artist(circle)
    plt.axis([bounds[0]-0.5, bounds[1]+0.5, bounds[0]-0.5, bounds[1]+0.5])
    plt.plot(start[0], start[1], "xr", markersize=10)
    plt.plot(goal[0], goal[1], "xb", markersize=10)
    plt.legend(('start', 'goal'), loc='upper left')
    plt.gca().set_aspect('equal')

def test_plot_scene():
    plt.figure()
    plot_scene(obstacles, start, goal)
    plt.tight_layout()

def test_run_rrt():
    np.random.seed(7)
    rrt = RRT(start=start,
            goal=goal,
            bounds=bounds,
            obstacle_list=obstacles)
    path_rrt = rrt.plan()

    plt.figure(figsize=(6,6))
    plot_scene(obstacles, start, goal)
    rrt.draw_graph()
    if path_rrt is None:
        print("No viable path found")
    else:
        plt.plot([x for (x, y) in path_rrt], [y for (x, y) in path_rrt], '-r')
    plt.tight_layout()

    max_iter_array = [10, 20, 100, 300]
    plt.figure(figsize=(14,5))
    goal_out_of_bound = np.array([100,100])
    for i in range(4):
        plt.subplot(1, 4, i+1)
        np.random.seed(9)
        rrt = RRT(start=start,
                goal=goal_out_of_bound,
                bounds=bounds,
                obstacle_list=[],
                goal_sample_rate=0.0, 
                max_iter=max_iter_array[i])
        path = rrt.plan()
        plot_scene([], start, goal_out_of_bound)
        rrt.draw_graph()
        plt.title('max_iter = {}'.format(max_iter_array[i]))
    plt.tight_layout()

def test_rrt_star():
    np.random.seed(7)
    rrt_star = RRTStar(start=start,
            goal=goal,
            bounds=bounds,
            obstacle_list=obstacles, max_iter=1000)
    path_rrt_star, min_cost = rrt_star.plan()
    print('Minimum cost: {}'.format(min_cost))

    # Check the cost
    def path_cost(path):
        return sum(np.linalg.norm(path[i] - path[i + 1]) for i in range(len(path) - 1))

    if path_rrt_star:
        print('Length of the found path: {}'.format(path_cost(path_rrt_star)))

    plt.figure(figsize=(6,6))
    plot_scene(obstacles, start, goal)
    rrt_star.draw_graph()
    if path_rrt_star is None:
        print("No viable path found")
    else:
        plt.plot([x for (x, y) in path_rrt_star], [y for (x, y) in path_rrt_star], '-r')
    plt.tight_layout()