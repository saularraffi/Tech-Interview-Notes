# Tech Interview Notes

## > Methodology

1. Understand the scenario and what the question is asking of you
2. Write down details of the question (unique information gives you hints to optimal solutions) 
3. Determine any assumptions you may be making and ask the interviewer any clarifying questions
4. Illustrate the problem and start with a generic example with no special cases and of sufficient size (most provided examples are too small, by about 50%)
5. Start with the brute force approach. Come up with the algorithm (without actually implementing) and complexity.
6. Optimize your algorithm
    - Can I try to solve this problem **```intuitively```** and see what I come up with?
    - Can I find an optimal solution based on **```details/constraints```** given by the question?
    - Am I doing **```repetative work```** that can be avoided?
    - Are there **```data structures```** that are better to use than others? (array/string, stack, queue, hash table, tree, etc.)
    - What is the **```base case```**? Can I work up from there? (Dynamic programming)
    - Can I perform any **```precomputation```** on the input? Will sorting help me?
    - Can I **```manipulate the input```** to help me? (ex. changing values in an input array)
    - Can I implement **```two-pointer / multi-pointer traversal```**? Will the pointers move out-to-in or in-to-out?
    - Should I use **```recursion or iteration```**? Can I implement a decision tree (recursive)?
    - Can I implement **```BFS or DFS```**?
    - Are there **```time vs. space tradeoffs```** that can be made to make one or the other more efficient?
    - Can I **```simplify the input```** temporarily to make it easier to solve the problem?
    - Is there a **```different parameter for success```** I can look for?
7. Walk through the steps of your algorithm and make sure it works, conceptually
8. Implement the algorithm
9. Test your code
    1. look through the code to make sure the code looks fine
    2. walk through the code with an example input
    3. focus on "hot spots" (parts of code that often cayse problem)
        - base cases in recursive code
        - integer division
        - null nodes in binary trees
        - the start and end of iteration through a linked list
    4. test edge cases

## > Recursive Algorithm Methodology

[YouTube - 5 Simple Steps for Solving Any Recursive Problem](https://www.youtube.com/watch?v=ngCos392W4w)

1. What's the simplest possible input?
2. Play around with examples and visualize
3. Relate hard cases to simpler cases
4. Generalize the pattern
5. Write code by combining recursive pattern with the base case(s)

## > Questions To Ask Interviewer

- does the input data structure contain any of the following?
    - negative numbers? 
    - floating points? 
    - nulls?
    - elements of different same data type?
- can the input data structure be empty or null?
- are there any duplicate elements in the input data structure?
- is the input data structure sorted? in ascending or descending order?
- can I manipulate the input data structure
- if the input is a tree, is it a BST? is it a balanced tree?
    - if I'm given a tree and a node, is the node guaranteed to be in the tree?
- if linked list, is it a singly or doubly linked list?
- if graph, are there any loops? any self-loops?
- should I optimize for space or time?
- how big is the size of the input? how big is the range of values?

## > Python Specific

### Strings and Arrays
- ```array[i], array[j] = array[j], array[y]``` --> swaps two values in array
- ```elem1, elem2 = array[0], array[1]``` --> in general you can assign 2 values at the same time
    - ```val1, val2 = dict[key1], dict[key2]``` --> works for dictionaries too
    - ```x, y = 1, 2``` --> even with just numbers
- ```list(string)``` --> turns string into array of chars and returns the array
    - ```list(dict)``` --> turns a dictionary into an array of keys and returns the array
    - ```list(dict.values())``` --> turns a dictionary into an array of the values and returns the array
- ```"".join(array)``` --> turn array of characters into string
- ```str.split(del)``` --> split string str by delimiter del into an array
- Dictionaries are ordered in python
- ```list.reverse()``` --> reverse an array, doesn't work on string
- ```list[::-1]``` --> reverse an array or string
- ```string[start:end]``` --> get substring from index start (inclusive) to index end (exclusive)
    - Also works for arrays
    - ```array[:x]``` --> from beginning to x (exclusive)
    - ```array[x:]``` --> from x (inclusive) to end
- ```f"My name is {saular} and I am {25} years old"``` --> this is an f string and it does the same things as the **.format()** method

### Hash Tables
- Sets are used to store multiple items in a single variable, ex. myset = {"apple", "banana", "cherry"}
- ```set.add(x)``` --> add an element to a set
- ```mySet = set()``` --> initialize empty set.  If you try to do mySet = {}, then you initialize a dictionary
- ```del dict[key]``` --> delete a key from dictionary
- ```dict.get(key, default)``` --> returns the value of the key, if key does not exist then return default
- ```set1.union(set2)``` --> combines set1 and set2 and returns resulting set (duplicates are removed)
- ```dict1.update(dict2)``` --> combines dict1 and dict2 and assings resulting dict to dict1 (duplicates are removed)
- ```mySet = {1,2,3} ... mySet.add(3)``` --> attempting to add a value that is already in a set will not add it to the set

### Iteration
- ```for key, value in dict.items():``` --> iterate over dictionary
- ```for i in range(x, -1, -1)``` --> iterate x times in reverse order
- ```for i in reversed(array)``` --> iterate over array in reverse
- ```[0 for col in range(n)]``` --> create an array of zeros with length n
- ```[[0 for col in range(n)] for row in range(m)]``` --> create an n by m sized matrix of zeros
- ```for idx, val in enumerate(array)``` --> allows you to loop over array and get both the value and the index, so now you can use this all the time rather than having to choose between the two types of for loops
- ```for _ in range(x)``` --> for when you don't need to access the variable in the for loop
- ```array.copy()``` --> returns a shallow copy of the array
    - ```array[:]``` --> returns the same thing, since the copy method may not work based on the python version

### Sorting
- ```sorted(array, key=lambda x: x[0])``` --> sort array by the first element of each sub array
- ```sorted(array)``` --> does the same thing as the sort() method but this returns the sorted array
    - ```sorted(string)``` --> takes in a string, splits up into array of characters, and sorts and returns the array of characters
- ```array.sort(reverse=True)``` sort an array in reverse order

### Math Operations
- ```%``` --> get remainder (ex. 10%3 = 1)
- If you take the modulo of a negative number, let's say -4%3, you get the number of 3's it takes to make -4 positive -> 2
- ```//``` --> round down division operator --> 5 / 3 = 2.5, but 5 // 3 = 2 --> eliminates need to cast to int
- ```num**x``` --> number to the power of x
- ```num % biggerNum = num``` --> ex. 8 % 10 = 8

### Other
- ```Ord(char)``` --> returns the Unicode value of the character char
- The sort method on arrays has time complexity O(nlog(n))
- ```x is not None``` --> check if x is not null (can't do x not None)
- ```if x is not None and ...``` --> so if x is None, then the second part of the if statement won't execute.  This is only the casing when ANDing
- in an if statement with an 'and' operator, both conditions execute. It's not that if the first is false, then the second won't execute
- ```float("inf") and float("-inf")``` --> infinity and negative infinity
- Only arrays, dictionaries, and sets are mutable in python
- When you pass a mutable datatype as an argument to python function, you can change its state within the function.  If you pass an immutable datatype as an argument to a python function, you cannot change its state
    - If you pass in an object, you can change its object properties, but you cannot assign the object itself to a new instance
- ```if type(x) is list``` --> check if element is a list; can work with other data types


## > General Notes

**Remember - you are not developing your skill to code, you are developing your skill to find the simplest and most optimal solution**

- When determining the worst-case space complexity of an algorithm, you consider the maximum space that needed to be used throughout the algorithm.  So, if 5 units of space are used on average but at a given point, n units of space are used, where n is the largest amount of space needed throughout the algorithm, then the space complexity is O(n).
- In most languages, concatenating to a string is O(n) operation, where n is the length of the string, whereas appending to a list is an O(1) operation.  This is because when concatenating to a string, python iterated through the string and adds the element(s) to the end of it --> O(n) complexity
- A problem that gives you a sorted array is a good indication that the problem can be solved in linear time or at least with a time complexity that is better than the brute force approach
- Look into tail recursion
- When deleting from linked list, assign the node to be deleted to a variable and after the previous node points to the appropriate node, set the node to be deleted to None
- When going through a while loop with 2 pointers on the array, consider where you may be able to implement a nested while loop that keeps moving one of the pointers
- Depth of a tree is the length of its longest branch (number of edges)
- Remember that if you are implementing a recursive algorithm, depending on how it is implemented, you can return a value at the end (will execute once all func calls are over)
- When analyzing the complexity of an algorithm, if there are two separate loops and they are looping over sets of different lengths, then take both into account --> ex.  algorithm loops over array1 (length n) then separately over array2 (length m), therefore time complexity is O(n + m)
- When analyzing the complexity of recursive tree-related algorithms, remember to consider the height of the tree.
- When implementing a recursive algorithm and there is information you need to store throughout the traversals, consider creating a class that holds the information you need and passing that object between function calls and then at the end, returning the desired attribute of the returning object. You can implement it the same exact way you would if you were using a dictionary.
- Breadth First Search --> use queue; Depth First Search --> use stack
    - Common theme for array based problems --> see what the result is up until that point, make a determination, continue
- 3-color depth first search --> to learn more, check out the **Cycle in Graph** problem
- the join operation to turn an array of strings into one string (ex. "".join(array)) has a time complexity of O(n)
- when you're planning on solving a problem recursively, at least know the benefits that an iterative solution would have offered --> easier to debug and no avoiding using up space on the call stack
- when implementing two pointer traversal, it's easier to move pointers from out to in rather than from in to out
- when you have multiple conditionals in an if statement, grouping statments into parentheses does matter
    - the statement ```if False and False or True``` is true, but the statement ```if False and (False or True)``` is false
        - this is because the first statement is interpreted as ```if (False and True) or True``` --> ```if False or True```
- There are 3 basic ways to represent a graph in memory (objects and pointers, matrix, and adjacency list) --> familiarize yourself with each representation and its pros & cons
- In general, a graph traveral algorithm will have a time complexity of **O(V + E)**, where ```V``` is the number of verticies and ```E``` is the number of edges
    - this is because some graphs have more verticies than edges (like when some nodes have no edges to/from other nodes) and some graphs have more edges than verticies

## > LeetCode Problems To Come Back To
### these are problems that had valuable lessons, not necessarily problems that were just hard

39. Combination Sum
547. Number of Provinces (popular in Amazon)

## > Code Notes

- when performing a recursive algorithm to generate some BST, this is valid
    ```python
    def generateTree(self, bst, array):
        # BASE CASE(S)
        # SOME CODE

        # bst initially passed in as None
        bst = TreeNode(...) # TreeNode is the BST class with val, left, and right properties
        bst.left = self.generateTree(bst.left, ...)
        bst.right = self.generateTree(bst.right, ...)
        
        return bst

    ```
    - because you first call the func with ```bst``` as ```None```, then you set ```bst``` to be a node, then you call the function on its left child and right child. In those function calls, ```bst``` is ```None``` again, and when they return, they return those trees to ```bst.left``` and ```bst.right```, respecively.

- consider using an enum for more readable code
    ```python
    HOME_TEAM_WON = 1
    AWAY_TEAM_WON = 2
    GAME_IS_TIE = 3

    # SOME CODE

    if result == HOME_TEAM_WON:
        # SOME CODE
    elif result == AWAY_TEAM_WON:
        # SOME CODE
    else:
        # SOME CODE
    ```

- binary search implemented
    ```python
    def binary_search(arr, target):
        start = 0
        end = len(arr) - 1
        mid = 0
    
        # remember <= and NOT <
        while start <= end:
            mid = (end + start) // 2
            if arr[mid] < target:
                start = mid + 1 # remember set to mid + 1, not mid
    
            elif arr[mid] > target:
                end = mid - 1 # remember set to mid - 1, not mid
    
            else:
                return mid
    
        return -1
    ```

- reverse an array in place
    ```python
    def reverseArrayInPlace(array):
        start = 0
        end = len(array) - 1
        while start < end:
            array[start], array[end] = array[end], array[start]
            start += 1
            end -= 1
    ```

- notice how the caching system is implemented in the ```nth fibonacci``` recursive algorithm
    ```python
    def getNthFib(n, cache={1: 0, 2: 1}):
        if n in cache:
            return cache[n]
        else:
            cache[n] = getNthFib(n - 1, cache) + getNthFib(n - 2, cache)
            return cache[n]
    ```

- setting up a for loop with reverse indexing
    ```python
    s = 'hello'

    for i in reversed(range(len(s))):
        print('i={} char={}'.format(i, s[i]))

    # this is the same as
    for i in reversed([0,1,2,3,4]):
        ...
    
    # and same as
    for i in [4,3,2,1,0]:
        ...

    # output
    # i=4 char=o
    # i=3 char=l
    # i=2 char=l
    # i=1 char=e
    # i=0 char=h
    ```

- example of using a class to manage data during a recursive algorithm
    ```python
    class TreeInfo:
        def __init__(self, nodeCount=0):
            self.nodeCount = nodeCount
    
    def getNodeCount(root):
        treeInfo = TreeInfo()
        getNodeCountHelper(root, treeInfo)
        return treeInfo.nodeCount
    
    def getNodeCountHelper(root, treeInfo):
        if root is None:
            return
        
        getNodeCountHelper(root.left, treeInfo)
        # working with TreeInfo
        treeInfo.nodeCount += 1 # <---
        getNodeCountHelper(root.right, treeInfo)
    ```

- another example of using a class to manage data during a recursive algorithm
    - this is from the ```Height Balanced Binary Tree``` problem from AlgoExpert, in which you must return true if the input tree is balanced and false if it is not
    ```python
    class TreeInfo:
        def __init__(self, height, isBalanced):
            self.height = height
            self.isBalanced = isBalanced
        
    def heightBalancedBinaryTree(tree):
        return heightBalancedBinaryTreeHelper(tree).isBalanced

    def heightBalancedBinaryTreeHelper(tree):
        if tree is None:
            return TreeInfo(0, True)
        
        leftSubtreeInfo = heightBalancedBinaryTreeHelper(tree.left)
        rightSubtreeInfo = heightBalancedBinaryTreeHelper(tree.right)
        
        # notice multi-line conditional --> better readability
        isBalanced = (
            leftSubtreeInfo.isBalanced and 
            rightSubtreeInfo.isBalanced and
            abs(leftSubtreeInfo.height - rightSubtreeInfo.height) <= 1
        )
        height = max(leftSubtreeInfo.height, rightSubtreeInfo.height) + 1
        
        return TreeInfo(height, isBalanced)
    ```

- a problem that has 2 scenarios --> ex. checking if array is monotonic (non-increasing or non-decreasing)
    ```python
    def isMonotonic(array):
        isNonDecreasing = True
        isNonIncreasing = True
        # you don't need to know whether to check fo non-increasing or non-decreasing
        # just attempt to disprove non-increasing and non-decreasing
        for i in range(1, len(array)):
            if array[i] < array[i - 1]:
                isNonDecreasing = False
            if array[i] < array[i - 1]:
                isNonIncreasing = False
        
        # if both disproven --> return False
        # if either were not disproven --> return True
        return isNonDecreasing or isNonIncreasing
    ```

- creating a BST without a helper function (for this example, assuming the array is sorted in ascending order)
    ```python
    def createTree(array):
        if len(array) == 0:
            return None

        mid = (len(array) - 1) // 2
        currentValue = array[mid]

        leftSubtree = createTree(array[:mid])
        rightSubtree = createTree(array[mid+1:])

        return TreeNode(currentValue, leftSubtree, rightSubtree)
    ```

- when a problem involves doing something to each level of a binary tree, consider an **iterative** algorithm using a **queue**
    - example, inverting a binary tree
    ```python
    def invertBinaryTree(tree):
        queue = [tree]
        # while queue is not empty
        while len(queue):
            current = queue.pop(0)
            if current is None:
                continue
            swapLeftAndRight(current)
            # queue allows for breadth first swapping
            # we don't move onto the next level until current level fully swapped
            queue.append(current.left)
            queue.append(current.right)

    def swapLeftAndRight(tree):
        tree.left, tree.right = tree.right, tree.left
    ```

- when passing an array to a recursive function, if you want the array to be exclusive to eachh call (rather than other function calls being able to manipulate the same array), then you must pass a **copy** of the array
    - this algorithm returns an array of all paths of a tree from its root to each of its lead nodes
    ```python
    # input
    #           2
    #         /   \
    #        1     3

    def getAllTreePaths(root):
        finalPaths = []
        getAllTreePathsHelper(root, [], finalPaths)
        return finalPaths

    def getAllTreePathsHelper(root, runningPath, finalPaths):
        if root is None:
            return

        runningPath.append(root.val)

        if root.left is None and root.right is None:
            finalPaths.append(runningPath)

        # see here that we are passing runningPath[:] to each function call, which passes a copy of runningPath
        getAllTreePathsHelper(root.left, runningPath[:], finalPaths)
        getAllTreePathsHelper(root.right, runningPath[:], finalPaths)

    # output
    # [[2, 1], [2, 3]]
    ```

    - this is what the output would be if the runningPath array itself, rather than a copy of it, was passed to each function call
    ```python
    # input
    #           2
    #         /   \
    #        1     3

    def getAllTreePaths(root):
        ...

    def getAllTreePathsHelper(root, runningPath, finalPaths):
        ...

        getAllTreePathsHelper(root.left, runningPath, finalPaths)
        getAllTreePathsHelper(root.right, runningPath, finalPaths)

    # output
    # [[2, 1, 3], [2, 1, 3]]
    ```
    - you can see that, because the same array was being manipulated throughout the function calls, the output subarrays are all the same

- you can embed a function definition within another function definition
    - this algorithm returns a unique list of all duplicate values within a Binary Tree
    ```python
    def getDuplicateValues(root):
        values = set()
        duplicates = set()

        # nested function definition
        def findDuplicates(root):
            if root is None:
                return
            if root.val in values and root.val not in duplicates:
                duplicates.add(root.val)

            values.add(root.val)
            
            findDuplicates(root.left)
            findDuplicates(root.right)

        # call the function here
        findDuplicates(root)
        return list(duplicates)
    ```
    - benefits of this
        - more readability --> you know that the nested function is only called by the parent function and its implementation can be abstracted from the overall implementation
        - you can use local variables within the parent function inside the nested function without passing them as arguments 

- example of an iterative approach to depth first search algorithm using stack (in this example, performing in-order traversal on a Binary Tree)
    ```python
    def inOrderTraversal(root):
        stack = []
        current = root

        while True:
            # keep pushing to stack while left child exists
            if current is not None:
                stack.append(current)
                current = current.left
            # after left checked, push right child to stack
            elif len(stack): # notice how we don't need to say 'elif len(stack) > 0'
                current = stack.pop()
                print(current.val)
                current = current.right
            else:
                break
    ```

- example of using a stack for DFS on a graph
    ```python
    def dfs(node):
        stack = [node]

        while len(stack):
            node = stack.pop()
            print(node.val)
            # we reverse so that the last child pushed onto the stack is the node's leftmost child
            for child in reversed(node.children):
                stack.append(child)
    ```

- sometimes you can use an array instead of a hash table (ex. when keeping track of visited elements in an array) because both will have a constant lookup time 
    - this algorithm check to see if an array completes one full cycle, treating each element in the array as the next index to check (not sure if this algorithm is implemented 100% correctly)
    ```python
    # input --> [4, 3, 1, 0, 2]
    # this contains one full cycle
    # [>4, 3, 1, 0, 2]
    # [4, 3, 1, 0, >2]
    # [4, 3, >1, 0, 2]
    # [4, >3, 1, 0, 2]
    # [4, 3, 1, >0, 2]
    # [>4, 3, 1, 0, 2]

    def completesOneFullCycle(array):
        idx = 0
        # every time an element is visited, that position is marked in positionsVisited
        positionsVisited = [False for _ in range(len(array))]
        for i in range((len(array))):
            nextIdx = array[idx]
            if positionsVisited[nextIdx] == True:
                return False
            idx = array[idx]
            positionsVisited[idx] = True

        return idx == 0
    ```

- heap class implementation
    ```python
    class MinHeap:
        def __init__(self, array):
            self.heap = self.buildHeap(array)
        
        # ===================== Core Functions =====================
        
        # O(n) time | O(1) space
        def buildHeap(self, array):
            for idx in reversed(range(len(array))):
                if not self.hasChildren(idx, array):
                    continue
                self.siftDown(idx, array)
            return array
        
        # O(log(n)) time | O(1) space
        def siftDown(self, currentIdx, heap):
            while self.hasChildren(currentIdx, heap):
                leftChildIdx = self.getLeftChildIdx(currentIdx, heap)
                rightChildIdx = self.getRightChildIdx(currentIdx, heap)
                smallerChildIdx = self.getSmallerChildIdx(currentIdx, heap)
                
                if heap[currentIdx] <= heap[smallerChildIdx]:
                    break
                
                self.swap(currentIdx, smallerChildIdx, heap)
                currentIdx = smallerChildIdx

        # O(log(n)) time | O(1) space
        def siftUp(self, currentIdx, heap):
            parentIdx = self.getParentIdx(currentIdx)
            
            while parentIdx != -1 and heap[currentIdx] < heap[parentIdx]:
                self.swap(currentIdx, parentIdx, heap)
                currentIdx = parentIdx
                parentIdx = self.getParentIdx(currentIdx)

        # O(1) time | O(1) space
        def peek(self):
            return self.heap[0]

        # O(log(n)) time | O(1) space
        def remove(self):
            lastIdx = len(self.heap) - 1
            self.swap(0, lastIdx, self.heap)
            removed = self.heap.pop()
            self.siftDown(0, self.heap)
            return removed

        # O(log(n)) time | O(1) space
        def insert(self, value):
            self.heap.append(value)
            lastIdx = len(self.heap) - 1
            self.siftUp(lastIdx, self.heap)
            
        # ===================== Helper Functions =====================

        def swap(self, i, j, heap):
            heap[i], heap[j] = heap[j], heap[i]
        
        def getParentIdx(self, idx):
            if idx == 0:
                return -1
            return (idx - 1) // 2
        
        def getLeftChildIdx(self, idx, heap):
            leftChildIdx = (idx * 2) + 1
            if leftChildIdx > len(heap) - 1:
                return -1
            return leftChildIdx
        
        def getRightChildIdx(self, idx, heap):
            rightChildIdx = (idx * 2) + 2
            if rightChildIdx > len(heap) - 1:
                return -1
            return rightChildIdx

        def hasChildren(self, idx, heap):
            leftChild = self.getLeftChildIdx(idx, heap)
            return leftChild != -1

        def getSmallerChildIdx(self, idx, heap):
            leftChildIdx = self.getLeftChildIdx(idx, heap)
            rightChildIdx = self.getRightChildIdx(idx, heap)
            if leftChildIdx == -1:
                return -1
            if rightChildIdx == -1:
                return leftChildIdx
            return leftChildIdx if heap[leftChildIdx] <= heap[rightChildIdx] else rightChildIdx
    ```