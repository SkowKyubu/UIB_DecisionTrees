There are two files : 
-main : corresponds to the implemented decision tree
-test : correspond to the tree from sklearn library

=== For the main file =================================

To make the program easy to test I implemented two functions ready to try :


print_result(max_depth, measure='accuracy', impurity_measure='entropy', print_tree=False, 
prune=False)

plot_result(max_depth, measure='accuracy')


measure can take this two values : 'accuracy' or 'time'
impurity_measure can take this two values : 'entropy' or 'gini'


