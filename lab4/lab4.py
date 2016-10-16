# MIT 6.034 Lab 4: Constraint Satisfaction Problems
# Written by Dylan Holmes (dxh), Jessica Noss (jmn), and 6.034 staff

from constraint_api import *
from test_problems import get_pokemon_problem

#### PART 1: WRITE A DEPTH-FIRST SEARCH CONSTRAINT SOLVER

def has_empty_domains(csp) :
    "Returns True if the problem has one or more empty domains, otherwise False"
    for var in csp.get_all_variables():
        if len(csp.get_domain(var)) < 1:
            return True
    return False

def check_all_constraints(csp) :
    """Return False if the problem's assigned values violate some constraint,
    otherwise True"""
    for cons in csp.get_all_constraints():
        var1_val, var2_val = map(csp.get_assigned_value, [cons.var1, cons.var2])
        if (var1_val != None) and (var2_val != None) and not cons.check(var1_val, var2_val):
            return False
    return True

def solve_constraint_dfs(problem) :
    """Solves the problem using depth-first search.  Returns a tuple containing:
    1. the solution (a dictionary mapping variables to assigned values), and
    2. the number of extensions made (the number of problems popped off the agenda).
    If no solution was found, return None as the first element of the tuple."""
    agenda = problem.unassigned_vars
    solution, ext = dfs(problem, agenda, 1)

    if solution:
        return solution.assigned_values, ext
    else:
        return None, ext

def dfs(csp, agenda, ext):
    if has_empty_domains(csp) or (not check_all_constraints(csp)):
        return None, ext
    if len(agenda) == 0:
        return csp, ext
    for value in csp.get_domain(agenda[0]):
        csp_ = csp.copy().set_assigned_value(agenda[0], value)
        csp_, ext = dfs(csp_, agenda[1:], ext+1)
        if csp_:
            return csp_, ext
    return None, ext

#### PART 2: DOMAIN REDUCTION BEFORE SEARCH

def eliminate_from_neighbors(csp_, var) :
    """Eliminates incompatible values from var's neighbors' domains, modifying
    the original csp.  Returns an alphabetically sorted list of the neighboring
    variables whose domains were reduced, with each variable appearing at most
    once.  If no domains were reduced, returns empty list.
    If a domain is reduced to size 0, quits immediately and returns None."""
    csp = csp_.copy()
    #print csp
    neighbors = csp.get_neighbors(var)
    reduced = set()
    for n in neighbors:
        if n not in reduced:
            for val2 in csp.get_domain(n):
                res = []
                for val1 in csp.get_domain(var):
                    res += [c.check(val1, val2) for c in csp.constraints_between(var, n)]
                #print n, val2, res
                if len(csp.constraints_between(var, n)) > 1 and not(all(res)):
                    csp_.eliminate(n, val2)
                    reduced.add(n)
                elif len(csp.constraints_between(var, n)) == 1 and all(not i for i in res):
                    csp_.eliminate(n, val2)
                    reduced.add(n)
                if len(csp_.get_domain(n)) == 0:
                    #print c, csp_.get_domain(n)
                    return None
    if len(reduced) == 0:
        return []
    return sorted(list(reduced))


def domain_reduction(csp, queue=None) :
    """Uses constraints to reduce domains, modifying the original csp.
    If queue is None, initializes propagation queue by adding all variables in
    their default order.  Returns a list of all variables that were dequeued,
    in the order they were removed from the queue.  Variables may appear in the
    list multiple times.
    If a domain is reduced to size 0, quits immediately and returns None."""
    queue = csp.get_all_variables() if queue==None else queue
    #print queue
    j = 0
    while j < len(queue):
        var = queue[j]
        reduced = eliminate_from_neighbors(csp, var)
        #print reduced
        if has_empty_domains(csp):
            return None
        j+=1
        check_no_repeat(queue, j, reduced)
    return queue

def from_list_to_set(list, set):
    for l in list:
        set.add(l)
    return set

def check_no_repeat(queue, j, reduced):
    queue_set = from_list_to_set(queue[j:], set())
    for i in reduced:
        if i not in queue_set:
            queue.append(i)

# QUESTION 1: How many extensions does it take to solve the Pokemon problem
#    with dfs if you DON'T use domain reduction before solving it?

# Hint: Use get_pokemon_problem() to get a new copy of the Pokemon problem
#    each time you want to solve it with a different search method.
solution, ext = solve_constraint_dfs(get_pokemon_problem())
ANSWER_1 = ext

# QUESTION 2: How many extensions does it take to solve the Pokemon problem
#    with dfs if you DO use domain reduction before solving it?
problem = get_pokemon_problem()
domain_reduction(problem)
solution, ext = solve_constraint_dfs(problem)
ANSWER_2 = ext


#### PART 3: PROPAGATION THROUGH REDUCED DOMAINS

def solve_constraint_propagate_reduced_domains(problem) :
    """Solves the problem using depth-first search with forward checking and
    propagation through all reduced domains.  Same return type as
    solve_constraint_dfs."""
    agenda = problem.unassigned_vars
    solution, ext = dfs_prop(problem, agenda, 1)
    if solution:
        return solution.assigned_values, ext
    else:
        return None, ext

def dfs_prop(csp, agenda, ext):
    if has_empty_domains(csp) or (not check_all_constraints(csp)):
        return None, ext
    if len(agenda) == 0:
        return csp, ext
    for value in csp.get_domain(agenda[0]):
        csp_ = csp.copy().set_assigned_value(agenda[0], value)
        res = domain_reduction(csp_, csp_.assigned_values.keys())
        csp_, ext = dfs_prop(csp_, agenda[1:], ext+1)
        if csp_:
            return csp_, ext
    return None, ext

# QUESTION 3: How many extensions does it take to solve the Pokemon problem
#    with propagation through reduced domains? (Don't use domain reduction
#    before solving it.)
solution, ext = solve_constraint_propagate_reduced_domains(get_pokemon_problem())
ANSWER_3 = ext


#### PART 4: PROPAGATION THROUGH SINGLETON DOMAINS

def domain_reduction_singleton_domains(csp, queue=None) :
    """Uses constraints to reduce domains, modifying the original csp.
    Only propagates through singleton domains.
    Same return type as domain_reduction."""
    queue = csp.get_all_variables() if queue==None else queue
    j = 0
    while j < len(queue):
        var = queue[j]
        reduced = eliminate_from_neighbors(csp, var)
        if has_empty_domains(csp):
            return None
        j+=1
        check_single_domain(queue, j, reduced, csp)
    return queue

def check_single_domain(queue, j, reduced, csp):
    queue_set = from_list_to_set(queue[j:], set())
    for i in reduced:
        if i not in queue_set and len(csp.get_domain(i)) == 1:
            queue.append(i)

def solve_constraint_propagate_singleton_domains(problem) :
    """Solves the problem using depth-first search with forward checking and
    propagation through singleton domains.  Same return type as
    solve_constraint_dfs."""
    agenda = problem.unassigned_vars
    solution, ext = dfs_singleton(problem, agenda, 1)
    if solution:
        return solution.assigned_values, ext
    else:
        return None, ext

def dfs_singleton(csp, agenda, ext):
    if has_empty_domains(csp) or (not check_all_constraints(csp)):
        return None, ext
    if len(agenda) == 0:
        return csp, ext
    for value in csp.get_domain(agenda[0]):
        csp_ = csp.copy().set_assigned_value(agenda[0], value)
        res = domain_reduction_singleton_domains(csp_, csp_.assigned_values.keys())
        csp_, ext = dfs_singleton(csp_, agenda[1:], ext+1)
        if csp_:
            return csp_, ext
    return None, ext

# QUESTION 4: How many extensions does it take to solve the Pokemon problem
#    with propagation through singleton domains? (Don't use domain reduction
#    before solving it.)
solution, ext = solve_constraint_propagate_singleton_domains(get_pokemon_problem())
ANSWER_4 = ext


#### PART 5: FORWARD CHECKING

def propagate(enqueue_condition_fn, csp, queue=None) :
    """Uses constraints to reduce domains, modifying the original csp.
    Uses enqueue_condition_fn to determine whether to enqueue a variable whose
    domain has been reduced.  Same return type as domain_reduction."""
    queue = csp.unassigned_vars if queue==None else queue
    j = 0
    while j < len(queue):
        var = queue[j]
        reduced = eliminate_from_neighbors(csp, var)
        if has_empty_domains(csp):
            return None
        j+=1
        for r in reduced:
            if enqueue_condition_fn(csp, r):
                queue_set = from_list_to_set(queue[j:], set())
                if r not in queue_set:
                    queue.append(r)
    return queue

def condition_domain_reduction(csp, var) :
    """Returns True if var should be enqueued under the all-reduced-domains
    condition, otherwise False"""
    return True

def condition_singleton(csp, var) :
    """Returns True if var should be enqueued under the singleton-domains
    condition, otherwise False"""
    return len(csp.get_domain(var)) == 1

def condition_forward_checking(csp, var) :
    """Returns True if var should be enqueued under the forward-checking
    condition, otherwise False"""
    return False


#### PART 6: GENERIC CSP SOLVER

def solve_constraint_generic(problem, enqueue_condition=None) :
    """Solves the problem, calling propagate with the specified enqueue
    condition (a function).  If enqueue_condition is None, uses DFS only.
    Same return type as solve_constraint_dfs."""
    if enqueue_condition==None:
        return solve_constraint_dfs(problem)
    agenda = problem.unassigned_vars
    solution, ext = dfs_generic(problem, agenda, 1, enqueue_condition)
    if solution:
        return solution.assigned_values, ext
    else:
        return None, ext

def dfs_generic(csp, agenda, ext, enqueue_condition):
    if has_empty_domains(csp) or (not check_all_constraints(csp)):
        return None, ext
    if len(agenda) == 0:
        return csp, ext
    for value in csp.get_domain(agenda[0]):
        csp_ = csp.copy().set_assigned_value(agenda[0], value)
        res = propagate(enqueue_condition, csp_, csp_.assigned_values.keys())
        csp_, ext = dfs_generic(csp_, agenda[1:], ext+1, enqueue_condition)
        if csp_:
            return csp_, ext
    return None, ext

# QUESTION 5: How many extensions does it take to solve the Pokemon problem
#    with DFS and forward checking, but no propagation? (Don't use domain
#    reduction before solving it.)
solution, ext = solve_constraint_generic(get_pokemon_problem(), condition_forward_checking)
ANSWER_5 = ext


#### PART 7: DEFINING CUSTOM CONSTRAINTS

def constraint_adjacent(m, n) :
    """Returns True if m and n are adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(m-n)==1

def constraint_not_adjacent(m, n) :
    """Returns True if m and n are NOT adjacent, otherwise False.
    Assume m and n are ints."""
    return abs(m-n)!=1

def all_different(variables) :
    """Returns a list of constraints, with one difference constraint between
    each pair of variables."""
    seen = set()
    res = []
    for v1 in variables:
        for v2 in variables:
            if (v1, v2) not in seen and (v2, v1) not in seen and v1 != v2:
                seen.add((v1, v2))
                seen.add((v2, v1))
                res+=[Constraint(v1,v2, constraint_different)]
    return res



#### PART 8: MOOSE PROBLEM (OPTIONAL)

moose_problem = ConstraintSatisfactionProblem(["You", "Moose", "McCain",
                                               "Palin", "Obama", "Biden"])

# Add domains and constraints to your moose_problem here:


# To test your moose_problem AFTER implementing all the solve_constraint
# methods above, change TEST_MOOSE_PROBLEM to True:
TEST_MOOSE_PROBLEM = False


#### SURVEY ###################################################

NAME = 'Chunchun Wu'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = '5'
WHAT_I_FOUND_INTERESTING = 'None'
WHAT_I_FOUND_BORING = 'None'
SUGGESTIONS = 'None'


###########################################################
### Ignore everything below this line; for testing only ###
###########################################################

if TEST_MOOSE_PROBLEM:
    # These lines are used in the local tester iff TEST_MOOSE_PROBLEM is True
    moose_answer_dfs = solve_constraint_dfs(moose_problem.copy())
    moose_answer_propany = solve_constraint_propagate_reduced_domains(moose_problem.copy())
    moose_answer_prop1 = solve_constraint_propagate_singleton_domains(moose_problem.copy())
    moose_answer_generic_dfs = solve_constraint_generic(moose_problem.copy(), None)
    moose_answer_generic_propany = solve_constraint_generic(moose_problem.copy(), condition_domain_reduction)
    moose_answer_generic_prop1 = solve_constraint_generic(moose_problem.copy(), condition_singleton)
    moose_answer_generic_fc = solve_constraint_generic(moose_problem.copy(), condition_forward_checking)
    moose_instance_for_domain_reduction = moose_problem.copy()
    moose_answer_domain_reduction = domain_reduction(moose_instance_for_domain_reduction)
    moose_instance_for_domain_reduction_singleton = moose_problem.copy()
    moose_answer_domain_reduction_singleton = domain_reduction_singleton_domains(moose_instance_for_domain_reduction_singleton)
