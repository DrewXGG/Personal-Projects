# Code for Programming Assignment 1, CSCI-UA.0472
#Yaoge Hu
import numpy as np

# An Element consists of a name (a string) and a cost (a number)

class Element:
    def __init__(self,name,cost):
        self.name = name
        self.cost = cost

# A CSet consists of a name, a set of Elements, 
# and a cost, which is the sum of the costs of the elements

class CSet:
    def __init__(self,name,elements):
        self.name = name
        self.elements = elements
        self.cost = 0
        for e in self.elements:
            self.cost += e.cost

def problem1():
    ea = Element('a',3)
    eb = Element('b',3)
    ec = Element('c',3)
    ed = Element('d',3)
    ee = Element('e',3)
    ef = Element('f',1)
    eg = Element('g',1)
    eh = Element('h',1)    
    s1 = CSet('S1',{ea,eb,ec,ed,ee})
    s2 = CSet('S2',{ea,ee,ef,eg,eh})
    s3 = CSet('S3',{eb,ec,eg,eh})
    s4 = CSet('S4',{eb,ef})
    s5 = CSet('S5',{ed,eh})
    s6 = CSet('S6',{ea,ed,ee,ef,eh})
    return {ea,eb,ec,ed,ee,ef,eg,eh}, [s1,s2,s3,s4,s5,s6]


## BestSetCoverIDS uses iterative deepening search to solve an instance of BESTSETCOVER
## problem is a pair (omega,collection) where omega is a set of Elements and collection is
## a list of CSets
## budget is the budget
## verbose is a Boolean flag, indicating the amount of printed output to generate
## BestSetCover should return two values: 
# 1. A solution, in the form of a list of CSets (a sublist of collection, in arbitrary order)
# 2. A Boolean 
# It should print out either just the solution, if verbose is false, or a trace of the execution
# if verbose is true

def BestSetCoverIDS(problem,budget,verbose):
## WRITE THE CODE FOR THIS
    omega, collection = problem()
    
    # max depth of the state space
    dpth = len(collection)
    
    # dfs function: return all legitimate combinations of sets(subset of collection) at the same depth
    def search(remain_omega, selected_sets, start_idx, cur_depth, max_depth, cur_cost):
        
        #base case: reach the depth and return each state's combination of sets at the maxdepth
        if cur_depth == max_depth:
            return [set(selected_sets)] 
        
        if cur_cost > budget:
            return []

        all_results = []  

        #find the rest of the sets for one state at the same depth
        for i in range(start_idx, len(collection)):
            cset = collection[i]

            
            new_remain_omega = remain_omega - cset.elements

            
            if cur_cost + cset.cost > budget:
                continue

            #add the set to the new sets if satisfy added cost < budget
            new_selected_sets = selected_sets + [cset]

            #update parameters and recursivly find the combination of sets
            results = search(new_remain_omega, new_selected_sets, i + 1, cur_depth + 1, max_depth, cur_cost + cset.cost)
            

            all_results.extend(results)
            
            
        return all_results  
    
    #each depth iteration
    for depth in range(1, dpth + 1):
        if verbose:
            print(f"Searching to depth {depth}")
            print("New States :")
        
        #dfs call
        all_states = search(omega, [], 0, 0, depth, 0)

    
        if not all_states:
            if verbose:
                print(f"Search terminated at depth {depth}")
            return set(), False

        
        #print states at each depth
        for state in all_states:
            state_trace = '{' + ', '.join(sorted(c.name for c in state)) + '}' 
            if verbose:
                print(f"State: {state_trace}")
            covered_elements = set()
            for cset in state:
                covered_elements = covered_elements.union(cset.elements)

            if covered_elements == omega:
                solution_names = sorted([c.name for c in state])
                if verbose:
                    print(f"Solution found at depth {depth}: {{{', '.join(solution_names)}}}")
                return state,True
    # if no soln the return set(),False
    return set(), False   
    





# testHC is a superroutine designed to facilitate debugging of BestSetCoverIDS. It has the
# extra argument "seed" which initializes the seed for the random number generator,
# so that you get replicable results

def testHC(problem,budget,ntries,verbose,seed):
    np.random.seed(seed)
    return BestSetCoverHillClimb(problem,budget,ntries,verbose)   

## BestSetCoverHillClimb uses hill climbing with random restart to solve the BEST SET COVER problem
## problem is a pair (omega,collection) where omega is a set of Elements and collection is
## a list of CSets
## budget is the budget
## verbose is a Boolean flag, indicating the amount of printed output to generate
## BestSetCoverHillClimd should return two values
# 1. A solution, in the form of a list of CSets (a sublist of collection, in arbitrary order)
# 2. A Boolean 
# It should print out either just the solution, if verbose is false, or a trace of the execution
# if verbose is true

def BestSetCoverHillClimb(problem,budget,ntries,verbose):
## WRITE THE CODE FOR THIS
    omega, collection = problem()
    
    state = []

    
    #error = the budget overrun + sum of costs of the items that are uncovered
    def error(omega,state,budget):
        overrun = 0
        remain = omega.copy()
        cost = 0
        remain_cost = 0
       
        #costs of unconvered items
        for cset in state:
            cost = cost + cset.cost
            remain = remain - cset.elements
            
        remain_cost = sum(element.cost for element in remain)
        
        #budget overrun
        if(cost-budget > 0):
            overrun = cost - budget         
        else: overrun = 0
        
        err = overrun + remain_cost
       
        return err
    
    #hillclimbing process
    for n in range(ntries):
        state = []
        #random start state: 50%
        for i in range(len(collection)):
            random_value = np.random.randint(2)
            if random_value == 0:
                continue
            else:
                state.append(collection[i]) 
        if verbose:
            start_name = [c.name for c in state]
            print(f"Attempt {n} : Starting State:{{{', '.join(start_name)}}}")
        
       
        
        #if verbose:
         #   print(f"New iteration.State:{state} Error: {err}")
          #  print('Neighbors:')
        start_state = state
         
        while(True):
            
            neighbors = []
    
            err = error(omega,start_state,budget)
            
            if verbose:
                start_name = [c.name for c in start_state]
                print(f"New iteration.State:{{{', '.join(start_name)}}} Error: {err}")
                print('Neighbors:')
            #finding neighbors    
            for cset in collection:
                if cset in start_state:
                    a = start_state.copy()
                    a.remove(cset)
                    neighbors.append(a)
                else:    
                    a = start_state.copy()
                    a.append(cset)
                    neighbors.append(a)
                    #start_state = state
           # if verbose:
              #  for nbor in neighbors:
                   # nbor_name =[c.name for c in nbor]
                   # print(f'State :{{{', '.join(nbor_name)}}} Error: {error(omega,nbor,budget)}')
            #choosing neighbor/return solution/entering a new attempt
            lowest_err = err
            for nbor in neighbors:
                if verbose:
                    nbor_name =[c.name for c in nbor]
                    print(f'State :{{{', '.join(nbor_name)}}} Error: {error(omega,nbor,budget)}')
                nbor_err = error(omega, nbor, budget)
                if nbor_err < lowest_err:
                    start_state = nbor
                    lowest_err = nbor_err
                if error(omega,start_state,budget) == 0:
                    sol_name = [c.name for c in start_state]
                    print(f"Solution: State{{{', '.join(sol_name)}}}")
                    return start_state,True
            if error(omega,start_state,budget) >= err:
                break 
            
    print("No solution found")
    return set(),False
#Testing
#BestSetCoverIDS(problem1, 30, True)
testHC(problem1,30,3,True,1)