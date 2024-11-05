import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.choice(data_length, data_length, replace=True))
            # Your Code Here
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            bag_idx = self.indices_list[bag]
            data_bag, target_bag = data[bag_idx], target[bag_idx] # Your Code Here
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        '''

        predictions = np.zeros((len(data), len(self.models_list)))
        for i, model in enumerate(self.models_list):
            predictions[:, i] = model.predict(data)
        return np.mean(predictions, axis=1)
        # Your code here
    
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        '''

        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        # # Your Code Here
        for i, x in enumerate(self.data):
            for j, model in enumerate(self.models_list):
                if i not in self.indices_list[j]:
                    list_of_predictions_lists[i].append(model.predict([x])[0])
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)

    
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        '''
        self._get_oob_predictions_from_every_model()
        # Your Code Here
        oob_predict = [ np.mean(predictions) if predictions else None for predictions in self.list_of_predictions_lists ]
        self.oob_predictions = np.array(oob_predict)
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        '''
        # Your Code Here
        self._get_averaged_oob_predictions()
        mask = np.array(self.oob_predictions) != None
        return np.mean((np.array(self.oob_predictions)[mask] - self.target[mask]) ** 2)