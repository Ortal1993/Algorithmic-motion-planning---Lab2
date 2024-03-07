import numpy as np
from environment import Environment
from kinematics import UR5e_PARAMS, Transform
from planners import RRT_STAR
from building_blocks import Building_Blocks
from visualizer import Visualize_UR


def main():
    ur_params = UR5e_PARAMS(inflation_factor=1)
    env = Environment(env_idx=2)
    transform = Transform(ur_params)
    
    bb = Building_Blocks(transform=transform, 
                        ur_params=ur_params, 
                        env=env,
                        resolution=0.1, 
                        p_bias=0.05,)
    
    rrt_star_planner = RRT_STAR(max_step_size=0.5,
                                max_itr=2000, 
                                bb=bb)
    
    visualizer = Visualize_UR(ur_params, env=env, transform=transform, bb=bb)
    
    # --------- configurations-------------
    env2_start = np.deg2rad([110, -70, 90, -90, -90, 0 ])
    env2_goal = np.deg2rad([50, -80, 90, -90, -90, 0 ])
    # ---------------------------------------
    filename = 'task2'
    path = rrt_star_planner.find_path(start_conf=env2_start,
                                        goal_conf=env2_goal,
                                        filename=filename)
                                                
    try:
        np.save(filename+'_path', path)
        path = np.load(filename+'_path.npy')
        visualizer.show_path(path)
        visualizer.show_conf(env2_goal)
    except:
        print('No Path Found')
    
    
   

if __name__ == '__main__':
    main()



