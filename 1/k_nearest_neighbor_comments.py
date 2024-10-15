import numpy as np
"""
Credits: the original code belongs to Stanford CS231n course assignment1. Source link: http://cs231n.github.io/assignments2019/assignment1/
"""

class KNearestNeighbor:
    """ a kNN classifier with L2 distance """

    def __init__(self):
        pass

    def fit(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y


        '''
        Эта функция fit используется для обучения классификатора методом k-ближайших соседей (k-NN). 
        Однако, в отличие от других алгоритмов машинного обучения, k-NN не требует сложных вычислений 
        во время обучения. Вместо этого, он просто запоминает обучающие данные.

        Функция принимает два входных параметра:

        X: numpy-массив размера (num_train, D), содержащий обучающие данные. Здесь num_train - количество обучающих примеров, 
        а D - размерность каждого примера.
        y: numpy-массив размера (N,), содержащий метки обучающих данных. Здесь y[i] - метка для i-го примера в X.
        Функция просто сохраняет эти данные в атрибутах объекта классификатора:

        self.X_train сохраняет обучающие данные X.
        self.y_train сохраняет метки обучающих данных y.
        Это все, что делает функция fit в данном случае. Поскольку k-NN не требует обучения в классическом смысле, 
        эта функция просто сохраняет данные для последующего использования при предсказании меток для новых данных.
        '''

    def predict(self, X, k=1, num_loops=0):
        """
        Predict labels for test data using this classifier.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.
        - k: The number of nearest neighbors that vote for the predicted labels.
        - num_loops: Determines which implementation to use to compute distances
          between training points and testing points.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self.predict_labels(dists, k=k)

    def compute_distances_two_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a nested loop over both the training data and the
        test data.

        Inputs:
        - X: A numpy array of shape (num_test, D) containing test data.

        Returns:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          is the Euclidean distance between the ith test point and the jth training
          point.
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            for j in range(num_train):
                #####################################################################
                # TODO:                                                             #
                # Compute the l2 distance between the ith test point and the jth    #
                # training point, and store the result in dists[i, j]. You should   #
                # not use a loop over dimension, nor use np.linalg.norm().          #
                #####################################################################
                # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

                # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists
        '''
        Эта функция compute_distances_two_loops вычисляет расстояние между каждой точкой тестовой выборки в X 
        и каждой точкой обучающей выборки в self.X_train используя вложенные циклы по обоим наборам данных.

        Функция принимает один входной параметр:

        X: numpy-массив размера (num_test, D) содержащий тестовые данные.
        Функция возвращает numpy-массив dists размера (num_test, num_train), где dists[i, j] представляет собой 
        евклидово расстояние между i-й точкой тестовой выборки и j-й точкой обучающей выборки.

        В функции используются два цикла:

        Внешний цикл по i пробегает все точки тестовой выборки.
        Внутренний цикл по j пробегает все точки обучающей выборки.
        В теле внутреннего цикла вычисляется евклидово расстояние между i-й точкой тестовой выборки и j-й точкой 
        обучающей выборки используя формулу:

        dists[i, j] = np.sqrt(np.sum(np.square(X[i] - self.X_train[j])))

        Эта формула вычисляет евклидово расстояние между двумя векторами X[i] и self.X_train[j].
        '''

    def compute_distances_one_loop(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using a single loop over the test data.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        for i in range(num_test):
            #######################################################################
            # TODO:                                                               #
            # Compute the l2 distance between the ith test point and all training #
            # points, and store the result in dists[i, :].                        #
            # Do not use np.linalg.norm().                                        #
            #######################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

        '''
        Эта функция compute_distances_one_loop вычисляет расстояние между каждой точкой тестовой выборки в X и каждой 
        точкой обучающей выборки в self.X_train используя один цикл по тестовой выборке.

        Функция принимает один входной параметр:

        X: numpy-массив размера (num_test, D) содержащий тестовые данные.
        Функция возвращает numpy-массив dists размера (num_test, num_train), где dists[i, j] представляет собой 
        евклидово расстояние между i-й точкой тестовой выборки и j-й точкой обучающей выборки.

        В функции используется один цикл по i, который пробегает все точки тестовой выборки.

        В теле цикла вычисляется евклидово расстояние между i-й точкой тестовой выборки и всеми точками обучающей выборки.

        В формуле:

        dists[i, :] = np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1))

        np.square(X[i] - self.X_train) вычисляет квадраты разности между i-й точкой тестовой выборки и каждой точкой 
        обучающей выборки, по элементам.

        np.sum(np.square(X[i] - self.X_train), axis=1) вычисляет сумму квадратов разности между i-й точкой тестовой 
        выборки и каждой точкой обучающей выборки, по элементам.

        np.sqrt(np.sum(np.square(X[i] - self.X_train), axis=1)) вычисляет евклидово расстояние между i-й точкой тестовой 
        выборки и каждой точкой обучающей выборки.
        '''

    def compute_distances_no_loops(self, X):
        """
        Compute the distance between each test point in X and each training point
        in self.X_train using no explicit loops.

        Input / Output: Same as compute_distances_two_loops
        """
        num_test = X.shape[0]
        num_train = self.X_train.shape[0]
        dists = np.zeros((num_test, num_train))
        #########################################################################
        # TODO:                                                                 #
        # Compute the l2 distance between all test points and all training      #
        # points without using any explicit loops, and store the result in      #
        # dists.                                                                #
        #                                                                       #
        # You should implement this function using only basic array operations; #
        # in particular you should not use functions from scipy,                #
        # nor use np.linalg.norm().                                             #
        #                                                                       #
        # HINT: Try to formulate the l2 distance using matrix multiplication    #
        #       and two broadcast sums.                                         #
        #########################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        dists = np.sqrt(np.sum(np.square(X), axis=1)[:, np.newaxis] + np.sum(np.square(self.X_train), axis=1) - 2 * np.dot(X, self.X_train.T))

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        return dists

        '''
        Функция принимает один входной параметр:

        X: numpy-массив размера (num_test, D) содержащий тестовые данные.
        Функция возвращает numpy-массив dists размера (num_test, num_train), 
        где dists[i, j] представляет собой евклидово расстояние между i-й точкой 
        тестовой выборки и j-й точкой обучающей выборки.

        В функции используется формула:

        dists = np.sqrt(np.sum(np.square(X), axis=1)[:, np.newaxis] + np.sum(np.square(self.X_train), axis=1) - 2 * np.dot(X, self.X_train.T))

        Эта формула вычисляет евклидово расстояние между всеми точками тестовой выборки и всеми точками обучающей выборки.

        В формуле используются следующие операции:

        np.sum(np.square(X), axis=1): вычисляет сумму квадратов элементов каждой строки в X.
        np.sum(np.square(self.X_train), axis=1): вычисляет сумму квадратов элементов каждой строки в self.X_train.
        np.dot(X, self.X_train.T): вычисляет скалярное произведение между каждой строкой в X и каждой строкой в self.X_train.
        np.newaxis: добавляет новое измерение к массиву, чтобы можно было выполнить операцию бродкастинга.
        + и -: выполняют элементно-элементные операции сложения и вычитания.

        '''

    def predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            #########################################################################
            # TODO:                                                                 #
            # Use the distance matrix to find the k nearest neighbors of the ith    #
            # testing point, and use self.y_train to find the labels of these       #
            # neighbors. Store these labels in closest_y.                           #
            # Hint: Look up the function numpy.argsort.                             #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
            #########################################################################
            # TODO:                                                                 #
            # Now that you have found the labels of the k nearest neighbors, you    #
            # need to find the most common label in the list closest_y of labels.   #
            # Store this label in y_pred[i]. Break ties by choosing the smaller     #
            # label.                                                                #
            #########################################################################
            # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
                    # Find the indices of the k nearest neighbors
            closest_y_indices = np.argsort(dists[i])[:k]
            
            # Get the labels of the k nearest neighbors
            closest_y = self.y_train[closest_y_indices]
            
            # Find the most common label among the neighbors
            y_pred[i] = np.bincount(closest_y).argmax()  # Choose the smallest label in case of a tie


            # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        return y_pred

'''
Эта функция predict_labels предсказывает метку для каждой точки тестовой выборки на основе расстояний между 
тестовыми точками и обучающими точками.

Функция принимает два входных параметра:

dists: numpy-массив размера (num_test, num_train), где dists[i, j] представляет собой расстояние между i-й точкой 
тестовой выборки и j-й точкой обучающей выборки.
k: целое число, которое представляет собой количество ближайших соседей, которые нужно рассматривать при предсказании метки.
Функция возвращает numpy-массив y_pred размера (num_test,), где y_pred[i] представляет собой предсказанную метку для i-й точки тестовой выборки.

В функции используется следующий алгоритм:

Для каждой точки тестовой выборки i находим индексы k ближайших соседей, используя функцию np.argsort.
Получаем метки ближайших соседей, используя индексы и массив self.y_train.
Находим наиболее распространенную метку среди ближайших соседей, используя функцию np.bincount. 
В случае равенства выбираем меньшую метку.
'''
