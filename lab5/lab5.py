# MIT 6.034 Lab 5: k-Nearest Neighbors and Identification Trees
# Written by Jessica Noss (jmn), Dylan Holmes (dxh), and Jake Barnwell (jb16)

from api import *
from data import *
import math
log2 = lambda x: math.log(x, 2)
INF = float('inf')

################################################################################
############################# IDENTIFICATION TREES #############################
################################################################################

def id_tree_classify_point(point, id_tree):
    """Uses the input ID tree (an IdentificationTreeNode) to classify the point.
    Returns the point's classification."""
    if id_tree.is_leaf():
        return id_tree.get_node_classification()
    return id_tree_classify_point(point, id_tree.apply_classifier(point))

def split_on_classifier(data, classifier):
    """Given a set of data (as a list of points) and a Classifier object, uses
    the classifier to partition the data.  Returns a dict mapping each feature
    values to a list of points that have that value."""
    mapping = {}
    for point in data:
        if classifier.classify(point) in mapping:
            mapping[classifier.classify(point)].append(point)
        else:
            mapping[classifier.classify(point)] = [point]
    return mapping

#### CALCULATING DISORDER

def branch_disorder(data, target_classifier):
    """Given a list of points representing a single branch and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the branch."""
    mapping = split_on_classifier(data, target_classifier)
    disorder = 0
    for a_type in mapping:
        num_points = len(mapping[a_type])
        disorder -= num_points/float(len(data)) * log2(num_points/float(len(data)))
    return disorder

def average_test_disorder(data, test_classifier, target_classifier):
    """Given a list of points, a feature-test Classifier, and a Classifier
    for determining the true classification of each point, computes and returns
    the disorder of the feature-test stump."""
    mapping = split_on_classifier(data, test_classifier)
    disorder = 0
    for a_type in mapping:
        disorder += branch_disorder(mapping[a_type], target_classifier) * len(mapping[a_type])/float(len(data))
    return disorder

## To use your functions to solve part A2 of the "Identification of Trees"
## problem from 2014 Q2, uncomment the lines below and run lab5.py:
#for classifier in tree_classifiers:
#    print classifier.name, average_test_disorder(tree_data, classifier, feature_test("tree_type"))


#### CONSTRUCTING AN ID TREE

def find_best_classifier(data, possible_classifiers, target_classifier):
    """Given a list of points, a list of possible Classifiers to use as tests,
    and a Classifier for determining the true classification of each point,
    finds and returns the classifier with the lowest disorder.  Breaks ties by
    preferring classifiers that appear earlier in the list.  If the best
    classifier has only one branch, raises NoGoodClassifiersError."""
    disorder = INF
    best = None
    for classifier in possible_classifiers:
        if average_test_disorder(data, classifier, target_classifier) < disorder:
            best = classifier
            disorder = average_test_disorder(data, classifier, target_classifier)
    if disorder == 1:
        raise NoGoodClassifiersError
    return best

## To find the best classifier from 2014 Q2, Part A, uncomment:
#print find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type"))


def construct_greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node=None):
    """Given a list of points, a list of possible Classifiers to use as tests,
    a Classifier for determining the true classification of each point, and
    optionally a partially completed ID tree, returns a completed ID tree by
    adding classifiers and classifications until either perfect classification
    has been achieved, or there are no good classifiers left."""
    if id_tree_node == None:
        id_tree_node = IdentificationTreeNode(target_classifier)
    greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node)
    return id_tree_node


def greedy_id_tree(data, possible_classifiers, target_classifier, id_tree_node):
    if branch_disorder(data, target_classifier) == 0:
        id_tree_node.set_node_classification(target_classifier.classify(data[0]))
        return
    if id_tree_node.get_classifier() == None:
        try:
            best = find_best_classifier(data, possible_classifiers, target_classifier)
        except NoGoodClassifiersError:
            return
        mapping = split_on_classifier(data, best)
        id_tree_node.set_classifier_and_expand(best, mapping)
    else:
        mapping = split_on_classifier(data, id_tree_node.get_classifier())
    branches = id_tree_node.get_branches()
    for node in branches:
        greedy_id_tree(mapping[node], possible_classifiers, target_classifier, branches[node])


## To construct an ID tree for 2014 Q2, Part A:
#print construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))

## To use your ID tree to identify a mystery tree (2014 Q2, Part A4):
tree_tree = construct_greedy_id_tree(tree_data, tree_classifiers, feature_test("tree_type"))
print tree_tree

## To construct an ID tree for 2012 Q2 (Angels) or 2013 Q3 (numeric ID trees):
#print construct_greedy_id_tree(angel_data, angel_classifiers, feature_test("Classification"))
#print construct_greedy_id_tree(numeric_data, numeric_classifiers, feature_test("class"))


#### MULTIPLE CHOICE
tree_classifiers2 = [\
    feature_test("has_leaves"),
    feature_test("leaf_shape"),
    feature_test("orange_foliage"),
]

ANSWER_1 = find_best_classifier(tree_data, tree_classifiers, feature_test("tree_type")).name
ANSWER_2 = find_best_classifier(tree_data, tree_classifiers2, feature_test("tree_type")).name
ANSWER_3 = "orange_foliage"

ANSWER_4 = [2,3]
ANSWER_5 = [3]
ANSWER_6 = [2]
ANSWER_7 = 2

ANSWER_8 = 'No'
ANSWER_9 = 'No'


################################################################################
############################# k-NEAREST NEIGHBORS ##############################
################################################################################

#### MULTIPLE CHOICE: DRAWING BOUNDARIES

BOUNDARY_ANS_1 = 3
BOUNDARY_ANS_2 = 4

BOUNDARY_ANS_3 = 1
BOUNDARY_ANS_4 = 2

BOUNDARY_ANS_5 = 2
BOUNDARY_ANS_6 = 4
BOUNDARY_ANS_7 = 1
BOUNDARY_ANS_8 = 4
BOUNDARY_ANS_9 = 4

BOUNDARY_ANS_10 = 4
BOUNDARY_ANS_11 = 2
BOUNDARY_ANS_12 = 1
BOUNDARY_ANS_13 = 4
BOUNDARY_ANS_14 = 4


#### WARM-UP: DISTANCE METRICS

def dot_product(u, v):
    """Computes dot product of two vectors u and v, each represented as a tuple
    or list of coordinates.  Assume the two vectors are the same length."""
    return sum(u[i]*v[i] for i in range(len(u)))

def norm(v):
    "Computes length of a vector v, represented as a tuple or list of coords."
    return sum(v[i]**2 for i in range(len(v)))**0.5

def euclidean_distance(point1, point2):
    "Given two Points, computes and returns the Euclidean distance between them."
    u = point1.coords
    v = point2.coords
    return sum((u[i]-v[i])**2 for i in range(len(v)))**0.5

def manhattan_distance(point1, point2):
    "Given two Points, computes and returns the Manhattan distance between them."
    u = point1.coords
    v = point2.coords
    return sum(abs(u[i]-v[i]) for i in range(len(v)))

def hamming_distance(point1, point2):
    "Given two Points, computes and returns the Hamming distance between them."
    u = point1.coords
    v = point2.coords
    return sum(u[i] != v[i] for i in range(len(v)))

def cosine_distance(point1, point2):
    """Given two Points, computes and returns the cosine distance between them,
    where cosine distance is defined as 1-cos(angle_between(point1, point2))."""
    u = point1.coords
    v = point2.coords
    x = float(dot_product(u,v))
    y = norm(u)*norm(v)
    return 1-x/y

#### CLASSIFYING POINTS

def get_k_closest_points(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns a list containing the k points
    from the data that are closest to the test point, according to the distance
    metric.  Breaks ties lexicographically by coordinates."""
    data = sorted(sorted(data, key = lambda y: y.coords), key = lambda x: distance_metric(x, point))
    return data[:k]


def knn_classify_point(point, data, k, distance_metric):
    """Given a test point, a list of points (the data), an int 0 < k <= len(data),
    and a distance metric (a function), returns the classification of the test
    point based on its k nearest neighbors, as determined by the distance metric.
    Assumes there are no ties."""
    ans = get_k_closest_points(point, data, k, distance_metric)
    classification = {}
    most_likely = None
    times_appear = 0
    for point in ans:
        if point.classification in classification:
            classification[point.classification]+=1
        else:
            classification[point.classification]=1
        if classification[point.classification] > times_appear:
            times_appear = classification[point.classification]
            most_likely = point.classification
    return most_likely

## To run your classify function on the k-nearest neighbors problem from 2014 Q2
## part B2, uncomment the line below and try different values of k:
#print knn_classify_point(knn_tree_test_point, knn_tree_data, 5, euclidean_distance)


#### CHOOSING k

def cross_validate(data, k, distance_metric):
    """Given a list of points (the data), an int 0 < k <= len(data), and a
    distance metric (a function), performs leave-one-out cross-validation.
    Return the fraction of points classified correctly, as a float."""
    correct = 0
    attempts = 0
    for i in range(len(data)-1):
        testing = data[i]
        training = data[:i]+data[i+1:]
        attempts += 1
        if knn_classify_point(testing, training, k, distance_metric) == testing.classification:
            correct +=1
    if knn_classify_point(data[-1], data[:len(data)-1], k, distance_metric) == data[-1].classification:
        correct +=1
    attempts += 1
    return float(correct)/float(attempts)


def find_best_k_and_metric(data):
    """Given a list of points (the data), uses leave-one-out cross-validation to
    determine the best value of k and distance_metric, choosing from among the
    four distance metrics defined above.  Returns a tuple (k, distance_metric),
    where k is an int and distance_metric is a function."""
    test_fn = [euclidean_distance, manhattan_distance, hamming_distance, cosine_distance]
    ratio = 0
    best_k = 0
    best_fn = None
    for k in range(len(data)):
        for fn in test_fn:
            value = cross_validate(data, k, fn)
            if value > ratio:
                ratio = value
                best_k = k
                best_fn = fn
    return (best_k, best_fn)

## To find the best k and distance metric for 2014 Q2, part B, uncomment:
print find_best_k_and_metric(knn_tree_data)


#### MORE MULTIPLE CHOICE

kNN_ANSWER_1 = 'Overfitting'
kNN_ANSWER_2 = 'Underfitting'
kNN_ANSWER_3 = 4

kNN_ANSWER_4 = 4
kNN_ANSWER_5 = 1
kNN_ANSWER_6 = 3
kNN_ANSWER_7 = 3

#### SURVEY ###################################################

NAME = 'Chunchun Wu'
COLLABORATORS = 'None'
HOW_MANY_HOURS_THIS_LAB_TOOK = '4'
WHAT_I_FOUND_INTERESTING = 'Good API Documentations'
WHAT_I_FOUND_BORING = 'None'
SUGGESTIONS = 'None'
