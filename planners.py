import numpy as np
import time
import math
from RRTTree import RRTTree

class RRT_STAR(object):
    def __init__(self, max_step_size, max_itr, bb):
        self.max_step_size = max_step_size
        self.max_itr = max_itr
        self.bb = bb
        self.tree = RRTTree(bb)

        #our
        self.edges_start_end = {}
        self.vertices_with_costs = []
        self.goal_id = -1
    
    def find_path(self, start_conf, goal_conf, filename):
        """Implement RRT-STAR - Return a path as numpy array"""
        
        i = 1
        start_id = self.add_vertex(start_conf)

        #######
        while i < self.max_itr:
            i += 1

            rand_conf = self.bb.sample(goal_conf)
            conf_near_id, conf_near = self.tree.GetNearestVertex(rand_conf)
            if conf_near_id == self.goal_id:
                continue
            conf_new = self.extend(conf_near, rand_conf)            
            if self.bb.local_planner(conf_near, conf_new):                
                conf_new_id = -1
                if np.array_equal(conf_new, goal_conf):
                    if (self.goal_id == -1):                
                        conf_new_id = self.add_vertex(conf_new)
                        dist = self.bb.edge_cost(conf_near, conf_new)
                        self.add_edge(conf_near_id, conf_new_id, dist)
                        self.goal_id = conf_new_id
                    else:
                        conf_new_id = self.goal_id
                else:
                    conf_new_id = self.add_vertex(conf_new)
                    dist = self.bb.edge_cost(conf_near, conf_new)
                    self.add_edge(conf_near_id, conf_new_id, dist)
                
                k_nearest_ids, _ = self.tree.GetKNN(conf_new, self.get_k_num(i))
                
                for potential_father_id in k_nearest_ids:
                    self.rewire(potential_father_id, conf_new_id)

                if conf_new_id != self.goal_id:
                    for potential_child_id in k_nearest_ids:
                        self.rewire(conf_new_id, potential_child_id)
            
            print(i)
            
        #constructing the plan from the goal to the start
        path, cost = self.get_shortest_path(self.goal_id)
        
        return np.array(path), cost
    
    def extend(self, x_near, x_random)-> np.array:
        '''
        Implement the Extend method
        @param x_near - Nearest Neighbor
        @param x_random - random sampled configuration
        return the extended configuration
        '''
        dist = np.linalg.norm(x_random - x_near)
        if self.max_step_size >= dist:
            return x_random            
        x_random = x_near + (((x_random - x_near)/dist) * self.max_step_size)

        return x_random       
    
    def rewire(self, x_potential_parent_id, x_child_id) -> None:
        '''
        Implement the rewire method
        @param x_potential_parent_id - candidte to become a parent
        @param x_child_id - the id of the child vertex
        return None
        '''
        # TODO
        if self.is_edge_new(x_child_id, x_potential_parent_id):
            if self.bb.local_planner(self.tree.vertices[x_potential_parent_id], self.tree.vertices[x_child_id]) :
                cost_new_edge = self.bb.edge_cost(self.tree.vertices[x_potential_parent_id], self.tree.vertices[x_child_id])
                cost_father = self.vertices_with_costs[x_potential_parent_id].cost
                cost_child = self.vertices_with_costs[x_child_id].cost
                if cost_new_edge + cost_father < cost_child:
                    #removing the new edge from edges_start_end
                    if self.tree.edges[x_child_id] in self.edges_start_end:
                        if x_child_id in self.edges_start_end[self.tree.edges[x_child_id]]:
                            self.edges_start_end[self.tree.edges[x_child_id]].remove(x_child_id)
                            if len(self.edges_start_end[self.tree.edges[x_child_id]]) == 0:
                                del self.edges_start_end[self.tree.edges[x_child_id]]
                    
                    self.add_edge(x_potential_parent_id, x_child_id, cost_new_edge)
                    self.propagate_cost_to_children(x_child_id)      

    def get_shortest_path(self, dest):
        '''
        Returns the path and cost from some vertex to Tree's root
        @param dest - the id of some vertex
        return the shortest path and the cost
        '''
        # TODO
        path = []

        curr_id = dest
        root_id = self.tree.GetRootID()
        if curr_id != -1:
            while (curr_id != root_id):
                path.append(self.vertices_with_costs[curr_id].conf)
                curr_id = self.tree.edges[curr_id]
                #print("curr id: ", curr_id)
            path.append(self.vertices_with_costs[root_id].conf)

        cost = 0.0
        if len(path) == 0:
            cost = math.inf
        for i in range(len(path) - 1):
            cost += self.bb.edge_cost(path[i], path[i + 1])
            #print("cost: ", cost)

        return path, cost
    
    def get_k_num(self, i):
        '''
        Determines the number of K nearest neighbors for each iteration
        '''
        if i < 300:
            k_num = 1
        elif 300 <= i < 600:
            k_num = 3
        elif 600 <= i < 1000:
            k_num=5
        elif 1000 <= i < 1500:
            k_num=6
        else:
            k_num = 7
        return k_num

    #our
    def is_edge_new(self, id_child, id_father):
        if (id_child == id_father):
            return False
        
        if (id_child in self.tree.edges):
            if self.tree.edges[id_child] == id_father:
                return False
        if (id_child in self.edges_start_end):
            if id_father in self.edges_start_end[id_child]:
                return False
        if (id_father in self.tree.edges):
            if self.tree.edges[id_father] == id_child:
                return False
        if (id_father in self.edges_start_end):
            if id_child in self.edges_start_end[id_father]:
                return False
        
        return True
    
    #our function
    def propagate_cost_to_children(self, id_father):
        if id_father in self.edges_start_end:
            ids_father = []
            ids_father.append(id_father)

            for id_father in ids_father:
                vertices_to_change = self.edges_start_end[id_father]
                for v_id in vertices_to_change:
                    cost_edge = self.bb.edge_cost(self.tree.vertices[id_father], self.tree.vertices[v_id])
                    cost_father = self.vertices_with_costs[id_father].cost
                    self.vertices_with_costs[v_id].set_cost(cost_edge + cost_father)
                    if v_id in self.edges_start_end and len(self.edges_start_end[v_id]) != 0:
                        ids_father.append(v_id)

    ##########
    def add_edge_start_end(self, sid, eid):
        if sid not in self.edges_start_end:
            self.edges_start_end[sid] = []
        self.edges_start_end[sid].append(eid)

    def add_vertex(self, conf):
        '''
        Add a state to the tree.
        @param state state to add to the tree.
        '''
        vid = self.tree.AddVertex(conf)
        self.vertices_with_costs.append(RRTVertex(conf=conf))
        return vid

    ##########
    def add_edge(self, sid, eid, edge_cost):
        '''
        Adds an edge in the tree.
        @param sid start state ID
        @param eid end state ID
        '''
        self.tree.AddEdge(sid, eid)            
        self.add_edge_start_end(sid, eid)
        self.vertices_with_costs[eid].set_cost(self.vertices_with_costs[sid].cost + edge_cost)

class RRTVertex(object):

    def __init__(self, conf, cost=0):

        self.conf = conf
        self.cost = cost

    def set_cost(self, cost):
        '''
        Set the cost of the vertex.
        '''
        self.cost = cost