import numpy as np
import pandas as pd

# convert link string to link list to handle saving's key, i.e. str(10, 6) to (10, 6)
def get_node(link):
    link = link[1:]
    link = link[:-1]
    nodes = link.split(',')
    return [int(nodes[0]), int(nodes[1])]

# determine if a node is interior to a route
def interior(node, route):
    try:
        i = route.index(node)
        # adjacent to depot, not interior
        if i == 0 or i == (len(route) - 1):
            label = False
        else:
            label = True
    except:
        label = False
    
    return label

# merge two routes with a connection link
def merge(route0, route1, link):
    if route0.index(link[0]) != (len(route0) - 1):
        route0.reverse()
    
    if route1.index(link[1]) != 0:
        route1.reverse()
        
    return route0 + route1

# sum up to obtain the total passengers belonging to a route
def sum_cap(route):
    sum_cap = 0
    for node in route:
        sum_cap += nodes.demand[node]
    return sum_cap

# determine 4 things:
# 1. if the link in any route in routes -> determined by if count_in > 0
# 2. if yes, which node is in the route -> returned to node_sel
# 3. if yes, which route is the node belongs to -> returned to route id: i_route
# 4. are both of the nodes in the same route? -> overlap = 1, yes; otherwise, no
def which_route(link, routes):
    # assume nodes are not in any route
    node_sel = list()
    i_route = [-1, -1]
    count_in = 0
    
    for route in routes:
        for node in link:
            try:
                route.index(node)
                i_route[count_in] = routes.index(route)
                node_sel.append(node)
                count_in += 1
            except:
                pass
                
    if i_route[0] == i_route[1]:
        overlap = 1
    else:
        overlap = 0
        
    return node_sel, count_in, i_route, overlap

# read node data in coordinate (x,y) format
path = "/home/panpan/vehicle-routing-problem-vrp-Clarke-Wright-Savings-Method/"
nodes = pd.read_csv(path+'input/demand.csv', index_col = 'node')
nodes.rename(columns={"distance to depot":'d0'}, inplace = True)
node_number = len(nodes.index) - 1
print(nodes.head())

# read pairwise distance
pw = pd.read_csv(path+'input/pairwise.csv', index_col = 'Unnamed: 0')
pw.index.rename('',inplace = True)

# calculate savings for each link
savings = dict()
for r in pw.index:
    for c in pw.columns:
        if int(c) != int(r):            
            a = max(int(r), int(c))
            b = min(int(r), int(c))
            key = '(' + str(a) + ',' + str(b) + ')'
            savings[key] = nodes['d0'][int(r)] + nodes['d0'][int(c)] - pw[c][r]

# put savings in a pandas dataframe, and sort by descending
sv = pd.DataFrame.from_dict(savings, orient = 'index')
sv.rename(columns = {0:'saving'}, inplace = True)
sv.sort_values(by = ['saving'], ascending = False, inplace = True)
print(sv.head())

# create empty routes
routes = list()

# if there is any remaining customer to be served
remaining = True

# define capacity of the vehicle
cap = 23

# record steps
step = 0

# get a list of nodes, excluding the depot
node_list = list(nodes.index)
node_list.remove(0)

# run through each link in the saving list
for link in sv.index:
    step += 1
    if remaining:

        print('step ', step, ':')

        link = get_node(link)
        node_sel, num_in, i_route, overlap = which_route(link, routes)

        # condition a. Either, neither i nor j have already been assigned to a route, 
        # ...in which case a new route is initiated including both i and j.
        if num_in == 0:
            if sum_cap(link) <= cap:
                routes.append(link)
                node_list.remove(link[0])
                node_list.remove(link[1])
                print('\t','Link ', link, ' fulfills criteria a), so it is created as a new route')
            else:
                print('\t','Though Link ', link, ' fulfills criteria a), it exceeds maximum load, so skip this link.')

        # condition b. Or, exactly one of the two nodes (i or j) has already been included 
        # ...in an existing route and that point is not interior to that route 
        # ...(a point is interior to a route if it is not adjacent to the depot D in the order of traversal of nodes), 
        # ...in which case the link (i, j) is added to that same route.    
        elif num_in == 1:
            n_sel = node_sel[0]
            i_rt = i_route[0]
            position = routes[i_rt].index(n_sel)
            link_temp = link.copy()
            link_temp.remove(n_sel)
            node = link_temp[0]

            cond1 = (not interior(n_sel, routes[i_rt]))
            cond2 = (sum_cap(routes[i_rt] + [node]) <= cap)

            if cond1:
                if cond2:
                    print('\t','Link ', link, ' fulfills criteria b), so a new node is added to route ', routes[i_rt], '.')
                    if position == 0:
                        routes[i_rt].insert(0, node)
                    else:
                        routes[i_rt].append(node)
                    node_list.remove(node)
                else:
                    print('\t','Though Link ', link, ' fulfills criteria b), it exceeds maximum load, so skip this link.')
                    continue
            else:
                print('\t','For Link ', link, ', node ', n_sel, ' is interior to route ', routes[i_rt], ', so skip this link')
                continue

        # condition c. Or, both i and j have already been included in two different existing routes 
        # ...and neither point is interior to its route, in which case the two routes are merged.        
        else:
            if overlap == 0:
                cond1 = (not interior(node_sel[0], routes[i_route[0]]))
                cond2 = (not interior(node_sel[1], routes[i_route[1]]))
                cond3 = (sum_cap(routes[i_route[0]] + routes[i_route[1]]) <= cap)

                if cond1 and cond2:
                    if cond3:
                        route_temp = merge(routes[i_route[0]], routes[i_route[1]], node_sel)
                        temp1 = routes[i_route[0]]
                        temp2 = routes[i_route[1]]
                        routes.remove(temp1)
                        routes.remove(temp2)
                        routes.append(route_temp)
                        try:
                            node_list.remove(link[0])
                            node_list.remove(link[1])
                        except:
                            #print('\t', f"Node {link[0]} or {link[1]} has been removed in a previous step.")
                            pass
                        print('\t','Link ', link, ' fulfills criteria c), so route ', temp1, ' and route ', temp2, ' are merged')
                    else:
                        print('\t','Though Link ', link, ' fulfills criteria c), it exceeds maximum load, so skip this link.')
                        continue
                else:
                    print('\t','For link ', link, ', Two nodes are found in two different routes, but not all the nodes fulfill interior requirement, so skip this link')
                    continue
            else:
                print('\t','Link ', link, ' is already included in the routes')
                continue

        for route in routes: 
            print('\t','route: ', route, ' with load ', sum_cap(route))
    else:
        print('-------')
        print('All nodes are included in the routes, algorithm closed')
        break

    remaining = bool(len(node_list) > 0)

# check if any node is left, assign to a unique route
for node_o in node_list:
    routes.append([node_o])

# add depot to the routes
for route in routes:
    route.insert(0,0)
    route.append(0)

print('------')
print(f'Routes found are:{routes} ')

