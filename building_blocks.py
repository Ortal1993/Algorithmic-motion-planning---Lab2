import numpy as np
from scipy.spatial.distance import cdist
from visualizer import Visualize_UR

X = 0
Y = 1
Z = 2
MIN_RESOLUTION = 3

class Building_Blocks(object):
    '''
    @param resolution determines the resolution of the local planner(how many intermidiate configurations to check)
    @param p_bias determines the probability of the sample function to return the goal configuration
    '''
    def __init__(self, transform, ur_params, env, resolution=0.1, p_bias=0.05):
        self.transform = transform
        self.ur_params = ur_params
        self.env = env
        self.resolution = resolution
        self.p_bias = p_bias
        self.cost_weights = np.array([0.4, 0.3 ,0.2 ,0.1 ,0.07 ,0.05])
        
        self.checked_confs = {} #dict[conf] = bool #dict[conf] = bool, true if there is a collision and false otherwise

        self.possible_joints_collision = []
        for i in range(len(self.ur_params.ur_links)):
            for j in range(i + 2, len(self.ur_params.ur_links)):
                self.possible_joints_collision.append((self.ur_params.ur_links[i], self.ur_params.ur_links[j]))
        
    def sample(self, goal_conf) -> np.array:
        """
        sample random configuration
        @param goal_conf - the goal configuration
        """
        #samples a value for the joint. If the random number is less than p_bias, 
        #it samples from the goal configuration for that joint. 
        #Otherwise, it samples randomly within the joint limits.    
        if np.random.uniform(0, 1) < self.p_bias:
            return goal_conf
        # With probability 1 - p_bias, sample randomly within joint limits
        conf = []
        for _, joint_limits in self.ur_params.mechamical_limits.items():            
            lower_limit, upper_limit = joint_limits[0], joint_limits[1]
            conf.append(np.random.uniform(lower_limit, upper_limit))
        return np.array(conf)
           
    def is_in_collision(self, conf) -> bool:
        """
        check for collision in given configuration, arm-arm and arm-obstacle
        return True if in collision
        @param conf - some configuration 
        """
        #visualizer = Visualize_UR(self.ur_params, self.env, self.transform, self)
        #visualizer.show_conf(conf)

        tupl_conf = tuple(conf)
        try:
            res = self.checked_confs[tupl_conf]
            return res
        except KeyError:
            #if it was not checked already - set the default value as True 
            self.checked_confs[tupl_conf] = True
        
        global_sphere_coords = self.transform.conf2sphere_coords(conf)

        sphere_radius_per_joint = self.ur_params.sphere_radius

        # arm - floor/window collision
        for joint, spheres in global_sphere_coords.items():
            if joint == 'shoulder_link':
                    continue
            if any (((sphere[2] - sphere_radius_per_joint[joint]) <= 0) or (sphere[0] + sphere_radius_per_joint[joint] > 0.4) for sphere in spheres):
                return True
        
        # arm - arm collision
        spheres_per_joint = {joint: np.array(global_sphere_coords[joint]) for joint in self.ur_params.ur_links}
        for joint_1, joint_2 in self.possible_joints_collision:
            robot_spheres_1 = [spheres for spheres in spheres_per_joint[joint_1][:, :3]]
            robot_spheres_2 = [spheres for spheres in spheres_per_joint[joint_2][:, :3]]
            distances = cdist(robot_spheres_1, robot_spheres_2)            
            sum_of_radii = sphere_radius_per_joint[joint_1] + sphere_radius_per_joint[joint_2]
            if np.any(distances <= sum_of_radii):
                return True
    
        # arm - obstacle collision 
        obstacles = self.env.obstacles #np.array
        if obstacles.size > 0:
            for joint in self.ur_params.ur_links:
                robot_spheres = [spheres for spheres in spheres_per_joint[joint][:, :3]]
                distances = cdist(robot_spheres, obstacles)
                sum_of_radii = sphere_radius_per_joint[joint] + self.env.radius
                if np.any(distances <= sum_of_radii):
                    return True      

        self.checked_confs[tupl_conf] = False
        return False
    
    def local_planner(self, prev_conf ,current_conf) -> bool:
        '''check for collisions between two configurations - return True if trasition is valid
        @param prev_conf - some configuration
        @param current_conf - current configuration
        '''
        # TODO 
        # Generate intermediate configurations between prev_conf and current_conf
        dist_prev_curr = np.linalg.norm(current_conf - prev_conf)
        num_intermediate_configs = max(int(np.ceil(dist_prev_curr / self.resolution)), MIN_RESOLUTION)
        intermediate_configs = np.linspace(prev_conf, current_conf, num_intermediate_configs)

        # Check for collisions in intermediate configurations
        return not any(self.is_in_collision(config) for config in intermediate_configs)
    
    def edge_cost(self, conf1, conf2):
        '''
        Returns the Edge cost- the cost of transition from configuration 1 to configuration 2
        @param conf1 - configuration 1
        @param conf2 - configuration 2
        '''
        return np.dot(self.cost_weights, np.power(conf1-conf2,2)) ** 0.5
    