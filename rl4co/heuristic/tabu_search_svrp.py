from tensordict.tensordict import TensorDict
import torch
import random

class TabuSearch_svrp:
    
    def __init__(self, td, tabu_len_min=30, tabu_len_max=50):
        self.td = td
        self.batch_size = self.td["demand"].size(0)
        self.num_customers = self.td["demand"].size(1)
        
        # for single instance
        self.distance_matrix = None
        self.tabu_len_min = tabu_len_min
        self.tabu_len_max = tabu_len_max
        self.tabu_len = 0
        self.best_solution = None
        self.best_cost = None
        
    def forward(self):
        "get solutions of all batch instances"
        batch_solutions = list()
        for i in range(self.batch_size):
            instance_i = self.td[i]
            best_solution, best_cost = self.forward_single(instance_i)
            batch_solutions.append(best_solution)
        return batch_solutions
    
    
    def get_distance_matrix(self, instance_td):
        # read pairwise distance: 
        self.num_point = instance_td["locs"].size(0)   
        # self.pair_wise = torch.zeros((num_customers, num_customers))
        customer_locs = instance_td["locs"]     # [num_point]
        loc_left_x = customer_locs[:, 0][..., None].repeat(1, self.num_point)  # [num_locs, num_locs]
        loc_left_y = customer_locs[:, 1][..., None].repeat(1, self.num_point)  # [num_locs, num_locs]
        loc_right_x = torch.transpose(loc_left_x, -1, -2)      # [num_locs, num_locs]
        loc_right_y = torch.transpose(loc_left_y, -1, -2)
        # [num_locs, num_locs]
        self.distance_matrix = torch.sqrt((loc_left_x - loc_right_x)**2 + (loc_left_y  - loc_right_y)**2)
        return self.distance_matrix
  
    def forward_single(self, instance_td: TensorDict, n_iters=2000):
        "get solution of an instance"
        solution = list()
        instance = instance_td
        self.distance_matrix = self.get_distance_matrix(instance_td)        # [num_p, num_p] ,symatric 
        self.best_sol = self.generate_greedy_solution(instance_td)
        self.best_reward = self.get_reward(self.best_sol)
        ##
        while n_iters > 0:
            self.tabu_len = random.randint(self.tabu_len_min, self.tabu_len_max)
            searched_nb_num = 0
            
        
            nb_structure_idx = random.randint(0, 3)
            while searched_nb_num < 4:
                # search in a nbhood
                # self.candidate = self.find_neighborhood(nb_structure_idx)
                self.candidate = self.find_neighborhood(0)
                candidate_reward = self.get_reward(self.candidate)
                
                if candidate_reward > self.best_reward:
                    self.best_sol = self.candidate
                    self.best_reward = candidate_reward
                    # increase tabu_length and continue search in this neighbor structure
                    self.tabu_len += 1
                    self.tabu_len = min(self.tabu_len, self.tabu_len_max)
                else:
                    # else decrease tabu_length and search next neighbot hood
                    self.tabu_len -= 1
                    self.tabu_len = max(self.tabu_len, self.tabu_len_min)
                    nb_structure_idx += 1
                    nb_structure_idx %= 4
                    searched_nb_num += 1

            n_iters -= 1
        return self.best_solution, self.best_cost
    
    def get_reward(self, solution):
        return 0.
    def find_neighborhood(self, nb_idx):
        
        assert nb_idx < 4, "wrong neighborhood structure index"
        if nb_idx == 0:
            nbs, moves = self.search_specific()
        # elif nb_idx == 1:
        #     nbs, moves = self.two_opt()
        # elif nb_idx == 2:
        #     nbs, moves = self.swap_oper()
        # else:
        #     nbs, moves = self.reallocate()

        # find best solution in nbs
        self.best_in_nbs = None
        for nb, move in nbs, moves:
            pass
        return self.best_in_nbs
    
    def is_valid(self, sol: list):
        '''
            check if visits every customer once
        '''
    def satisfy_demand(self, route: list):
        '''
            check if route satisfy demand: not exceed capacity
        '''
        return True
        
    def search_specific(self):
        '''
        random select a customer in route l, insert to another route that can satisfy constraint.
        loop for every route l
        return:
            moves: nested list, (idx of seleted customer route, selected_customer, idx of insert customer route)
        '''
        neighborhood = []
        moves = []
        move = None 
        # iterate every route to selet and insert
        for i in range(len(self.best_sol)):
            selected_customer = random.choice(self.best_sol[i][1:-1])
            
            # iterate other all routes until choose a valid insert
            sol = [route[:] for route in self.best_sol]
            j = 0
            while j < len(self.best_sol):
                if j != i:      # can not be selected route
                    curr_route = self.best_sol[j][:]
                    # insert successfully then break
                    
                    curr_route.insert(-1, selected_customer)
                    if self.satisfy_demand(curr_route):
                        # return new solution
                        sol[i].remove(selected_customer)
                        sol[j] = curr_route
                        move = (i, selected_customer, j)
                        break
                    # else continue
                    j += 1
                    assert j < len(self.best_sol), "none of a route can add this customer"
                else:
                    j += 1
                    
            # check sol valid
            assert self.is_valid(sol), "this solution is not valid"
            # record move and nb
            neighborhood.append(sol)
            moves.append(move)
        
        return neighborhood, moves

    def two_opt(self):
        pass
    
    def generate_greedy_solution(self, instance_td):
        """greedy initial solution for an instance """
        customers = [i for i in range(1, self.num_customers+1)]
        solution = list()

        while len(customers) > 0:
            route = list()
            curr_point = 0
            used_capacity = torch.zeros(self.num_customers, device=instance_td.device)
            next_capacity = torch.zeros(self.num_customers, device=instance_td.device)
            # get satisfied demands customers from rest customers            
            
            # must in rest customer 
            mask = torch.ones((self.num_customers + 1), dtype=torch.bool, device=instance_td.device)
            mask[customers] = False
            negative = 1e3 * torch.ones_like(mask)
            while True:
                
                # must not exceed capacity
                next_capacity = used_capacity + instance_td["demand"]
                mask2 = next_capacity > 1.
                mask = mask | torch.cat((torch.ones(1, dtype=torch.bool, device=instance_td.device), mask2))
                distance = self.distance_matrix[curr_point]     # [num_cus]
                # True —— -1
                distance = torch.where(mask, negative, distance)
                
                if (distance > 1e2).all():       # no satisfied customers
                    solution.append(route)
                    break
                # get closet customer from satisfied ones
                closest = distance.argmin()     # from 0 to num_customer
                route.append(closest)
                used_capacity += instance_td["demand"][closest - 1]
                curr_point = closest
                customers.remove(closest)
                # change mask
                mask[closest] = True
            
        # add depot to start and end
        for route in solution:
            route.insert(0,0)
            route.append(0)
        
        return solution
