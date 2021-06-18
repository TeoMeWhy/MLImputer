import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class MLImputer(BaseEstimator, TransformerMixin):

    def __init__(self, variables: list, estimator: BaseEstimator, features:list):
        self.variables = variables  # variáveis para realizar a imputação
        self.estimator = estimator  # Tipo de algoritmo para realizar a imputação
        self.features = features
        self.estimators = {}

    def get_params(self, deep=False):
        return { 'variables': self.variables,
                 'estimator': self.estimator,
                 'features': self.features, }

    def fit(self, X:pd.DataFrame, y:pd.Series):
        '''
        Função que ajusta todos os modelos necessários para as variáveis que devem realizar imputação de dados.
        
        X: pd.DataFrame com todas a features do modelo final;
        y: pd.Series com a variável alvo para predição final
        '''
        
        values = X[self.features].count()
        
        self.features = values[values == X[self.features].shape[0]].index.tolist()

        for v in self.variables:
            self.fit_one(X = X, target = v)

        return self

    def transform(self, X:pd.DataFrame)->pd.DataFrame:
        '''
        Percorre todas as variáveis realizando imputação de dados
        
        X: pd.DataFrame com as features do modelo final
        '''

        for v in self.variables:
            X = self.transform_one(X, v)
        return X

    def fit_one(self, X: pd.DataFrame, target: str)->None:
        '''
        Ajusta um modelo para variável que necessita de imputação
        
        X: pd.DataFrame com apenas as variáveis do modelo final
        target: str com o nome da variável que devemos realizar a imputação de dados
        '''

        X_complete = X[~X[target].isna()].copy()

        model = self.estimator()
        model.fit(X_complete[self.features], X_complete[target])

        self.estimators[target] = model

        return None

    def transform_one(self, X:pd.DataFrame, target:str)->pd.DataFrame:
        '''
        Realiza a imputação de uma única variável
        
        X: pd.DataFrame com as variáveis do modelo final
        target: str nome da variável que necessita de imputação
        '''

        X_full = X[~X[target].isna()].copy()
        if X_full.shape[0] == X.shape[0]:
            return X

        X_na = X[X[target].isna()].copy()
        model = self.estimators[target]
        X_na[target] = model.predict(X_na[self.features])
        X_new = pd.concat([X_full, X_na]).sort_index()

        return X_new
