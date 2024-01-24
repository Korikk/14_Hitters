import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import re
from helpers.data_prep import *
from helpers.eda import *
from helpers.helpers import *

pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 170)

df = pd.read_pickle("Datasets/prepared_hitters_df.pkl")
check_df(df)

df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

cols = []
count = 1
for column in df.columns:
    if column == 'RBWAL':
        cols.append(f'RBWAL_{count}')
        count += 1
        continue
    cols.append(column)
df.columns = cols

y = df["SALARY"]
x = df.drop("SALARY", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=12, test_size=20)
lgb_model = LGBMRegressor().fit(x_train, y_train)

y_pred = lgb_model.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))


def plot_importance(model, features, num=len(x), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')


plot_importance(lgb_model, x_train)

#################
lgb_model_grid = LGBMRegressor()

# hyperparameters
lgb_params = {'max_depth': range(1, 11), "min_samples_split": [1, 2, 3, 4]}

lgb_cv = GridSearchCV(lgb_model_grid, lgb_params, cv=5, n_jobs=-1, verbose=True)
lgb_cv.fit(x_train, y_train)

lgb_cv.best_params_

lgb_tuned = LGBMRegressor(**lgb_cv.best_params_).fit(x_train, y_train)

# train error
y_pred = lgb_tuned.predict(x_test)
np.sqrt(mean_squared_error(y_test, y_pred))


plot_importance(lgb_tuned, x_train)