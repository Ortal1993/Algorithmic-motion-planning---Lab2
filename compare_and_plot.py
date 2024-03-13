import numpy as np
import matplotlib.pyplot as plt
import time as Time
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR
import cProfile
import os
import math

MAX_ITER = 2000

def main():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)

    num_runs = 5

    p_biases = [0.2]
    max_step_sizes = [0.1, 0.3, 0.5, 0.8, 1.2, 1.7, 2.0]

    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])

    #in_collision_floor = np.deg2rad([90, -50, 80, -110, -90, -10])
    #in_collision_arm = np.array([-0.694, -1.376, 2.55, -1.122, -1.570, 2.26])

    

    # Initialize empty dictionaries to store results
    results = {p_bias: {max_step_size: {'path': [], 'cost': [], 'time': []} for max_step_size in max_step_sizes} for p_bias in p_biases}
    
    filename_str = "results"

    # Run tests
    for p_bias in p_biases:
        bb = Building_Blocks(transform=transform, 
                        ur_params=ur_params, 
                        env=env,
                        resolution=0.1, 
                        p_bias=p_bias)       
        for max_step_size in max_step_sizes:
            #visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
            paths = []
            costs = []
            times = []
            for run in range(num_runs):               
                start_time = Time.time()

                rrt_star_planner = RRT_STAR(max_step_size=max_step_size,
                                max_itr=MAX_ITER, 
                                bb=bb)

                path, cost = rrt_star_planner.find_path(env2_start, env2_goal, filename_str)

                """try:
                    directory = "results"

                    # Create the directory if it doesn't exist
                    if not os.path.exists(directory):
                        os.makedirs(directory)

                    filename = filename_str + '_' + str(p_bias) + '_' + str(max_step_size) + '_' + str(run)
                    file_path = os.path.join(directory, filename)
                    np.save(file_path, path)
                    path = np.load(file_path+'.npy')
                    visualizer.show_path(path)
                except:
                    print('No Path Found')"""

                total_time = Time.time() - start_time

                # Record results
                paths.append(path)
                costs.append(cost)
                times.append(total_time)

            # Store average results
            results[p_bias][max_step_size]['path'] = paths
            results[p_bias][max_step_size]['cost'] = costs
            results[p_bias][max_step_size]['time'] = times

    print("results:", results)

    #find the best path from all paths
    all_results = [(results[p_bias][max_step_size]['cost'][i], results[p_bias][max_step_size]['path'][i])
               for p_bias in p_biases
               for max_step_size in max_step_sizes
               for i in range(num_runs)]

    # Find the tuple with the lowest cost
    lowest_cost_tuple = min(all_results, key=lambda x: x[0])

    # Extract the lowest cost and corresponding path
    lowest_cost = lowest_cost_tuple[0]
    best_path = lowest_cost_tuple[1]

    print(f"The best path is {best_path} with cost of {lowest_cost}")
    
    mean_results = {p_bias: {max_step_size: {'mean_cost': [], 'mean_time': []} for max_step_size in max_step_sizes} for p_bias in p_biases}
    for p_bias in p_biases:
        for max_step_size in max_step_sizes:
            sum_cost = 0
            sum_time = 0
            count = 0
            for i in range(num_runs):
                if results[p_bias][max_step_size]['cost'][i] != math.inf:
                    sum_cost += results[p_bias][max_step_size]['cost'][i]
                    sum_time += results[p_bias][max_step_size]['time'][i]
                    count += 1
            if count != 0:
                mean_results[p_bias][max_step_size]['mean_cost'] = sum_cost/count
                mean_results[p_bias][max_step_size]['mean_time'] = sum_time/count
            else: 
                mean_results[p_bias][max_step_size]['mean_cost'] = 0
                mean_results[p_bias][max_step_size]['mean_time'] = 0

    # Plot mean results
    for p_bias in p_biases:
        plt.figure()
        plt.title(f'Average cost as a function of computation time for p_bias={p_bias} \n for different step sizes')
        plt.xlabel('Computation Time (s)')
        plt.ylabel('Cost')
        for max_step_size in max_step_sizes:
            cost = mean_results[p_bias][max_step_size]['mean_cost']
            time = mean_results[p_bias][max_step_size]['mean_time']
            plt.scatter(time, cost, label=f'max_step_size={max_step_size}')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot results
    for p_bias in p_biases:
        plt.figure()
        plt.title(f'Cost as a function of computation time for p_bias={p_bias} \n for different step sizes')
        plt.xlabel('Computation Time (s)')
        plt.ylabel('Cost')
        for max_step_size in max_step_sizes:
            cost = results[p_bias][max_step_size]['cost']
            time = results[p_bias][max_step_size]['time']
            plt.scatter(time, cost, label=f'max_step_size={max_step_size}')
        plt.legend()
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    cProfile.run("main()", filename="profile_results.prof")
