import unittest
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#all tes cases
class TestDecisionTreeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = pd.read_csv('iris.csv')
        cls.model = joblib.load('model.joblib')

        train, test = train_test_split(cls.data, test_size=0.4, stratify=cls.data['species'], random_state=42)
        cls.X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
        cls.y_test = test['species']
    #test11
    def test_model_prediction(self):
        predictions = self.model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
    #test22
    def test_model_accuracy(self):
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        self.assertGreaterEqual(acc, 0.7)

if __name__ == '__main__':
    unittest.main()
#just adding comments
