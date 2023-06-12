from src.drawing import show_forecast_test_plot, show_time_series_only, show_acf_pacf
from sklearn.model_selection import train_test_split
from prophet import Prophet
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import lightgbm as lgb
from pmdarima.arima import auto_arima
import xgboost as xgb


def my_prophet(ts, train_size):
    time = ts.Date
    target = ts.target_value

    data = []
    for i in range(17):
        train_1, test_1 = train_test_split(ts, train_size=train_size + 7 * i, shuffle=False)
        train_dataset = pd.DataFrame()
        train_dataset['ds'] = train_1.Date
        train_dataset['y'] = train_1.target_value
        train_dataset['volume'] = train_1.Volume

        m = Prophet()
        m.fit(train_dataset)
        future = m.make_future_dataframe(periods=7)
        forecast = m.predict(future)
        data = data + list(forecast[-7:].yhat)

    return data


def my_xgboost(ts, train_size):
    def create_date_features(df):
        df['month'] = df.Date.dt.month
        df['day_of_month'] = df.Date.dt.day
        df['day_of_year'] = df.Date.dt.dayofyear
        # 1.1.2013 is Tuesday, so our starting point is the 2nd day of week
        df['day_of_week'] = df.Date.dt.dayofweek + 1
        df['year'] = df.Date.dt.year
        df["is_wknd"] = df.Date.dt.weekday // 4
        df['is_month_start'] = df.Date.dt.is_month_start.astype(int)
        df['is_month_end'] = df.Date.dt.is_month_end.astype(int)
        return df

    ts = create_date_features(ts)

    def lag_features(dataframe, lags):
        dataframe = dataframe.copy()
        for lag in lags:
            dataframe['lag_' + str(lag)] = dataframe['target_value'].transform(
                lambda x: x.shift(lag))
        return dataframe

    ts = lag_features(ts, [1, 2, 3, 4, 5, 7, 10])

    def roll_mean_features(dataframe, windows):
        dataframe = dataframe.copy()
        for window in windows:
            dataframe['mean_' + str(window)] = dataframe['target_value'].transform(lambda x:
                                                                                   x.shift(1).rolling(window=window,
                                                                                                      min_periods=1,
                                                                                                      win_type="triang").mean())
        return dataframe

    ts = roll_mean_features(ts, [1, 2, 7, 14, 30])

    def ewm_features(dataframe, alphas, lags):
        dataframe = dataframe.copy()
        for alpha in alphas:
            for lag in lags:
                dataframe['ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                    dataframe['target_value']. \
                        transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
        return dataframe

    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [1, 2, 3, 4, 5, 7, 10]
    ts = ewm_features(ts, alphas, lags)
    ts = pd.get_dummies(ts, columns=['day_of_week', 'month'])

    ts['target_value'] = np.log1p(ts["target_value"].values)

    time = ts.Date
    target = ts.target_value

    data = []
    for i in range(17):
        train, test = train_test_split(ts, train_size=train_size + 7 * i, shuffle=False)

        x_train = train.drop(['Date', 'target_value'], axis=1)
        y_train = train.target_value
        x_test = test.drop(['Date', 'target_value'], axis=1)
        y_test = test.target_value
        x_val = x_train[-4 * 7:]
        y_val = y_train[-4 * 7:]

        reg = XGBRegressor(n_estimators=100)
        reg.fit(x_train, y_train,
                eval_set=[(x_train, y_train), (x_val, y_val)],
                early_stopping_rounds=50,
                verbose=False)  # Change verbose to True if you want to see it train
        '''
        XGBRegressor(base_score=0.5, booster=None, colsample_bylevel=1,
               colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
               importance_type='gain', interaction_constraints=None,
               learning_rate=0.300000012, max_delta_step=0, max_depth=6,
               min_child_weight=1, missing=nan, monotone_constraints=None,
               n_estimators=100, n_jobs=0, num_parallel_tree=1,
               objective='reg:squarederror', random_state=0, reg_alpha=0,
               reg_lambda=1, scale_pos_weight=1, subsample=1, tree_method=None,
               validate_parameters=False, verbosity=None)
        '''

        forecast = reg.predict(x_test[:7])
        forecast = np.expm1(forecast)
        data += list(forecast)

    return data


def my_lgbm(ts, train_size):
    def create_date_features(df):
        df['month'] = df.Date.dt.month
        df['day_of_month'] = df.Date.dt.day
        df['day_of_year'] = df.Date.dt.dayofyear
        # 1.1.2013 is Tuesday, so our starting point is the 2nd day of week
        df['day_of_week'] = df.Date.dt.dayofweek + 1
        df['year'] = df.Date.dt.year
        df["is_wknd"] = df.Date.dt.weekday // 4
        df['is_month_start'] = df.Date.dt.is_month_start.astype(int)
        df['is_month_end'] = df.Date.dt.is_month_end.astype(int)
        return df

    ts = create_date_features(ts)

    def lag_features(dataframe, lags):
        dataframe = dataframe.copy()
        for lag in lags:
            dataframe['lag_' + str(lag)] = dataframe['target_value'].transform(
                lambda x: x.shift(lag))
        return dataframe

    ts = lag_features(ts, [1, 2, 3, 4, 5, 7, 10])

    def roll_mean_features(dataframe, windows):
        dataframe = dataframe.copy()
        for window in windows:
            dataframe['mean_' + str(window)] = dataframe['target_value'].transform(lambda x:
                                                                                   x.shift(1).rolling(window=window,
                                                                                                      min_periods=1,
                                                                                                      win_type="triang").mean())
        return dataframe

    ts = roll_mean_features(ts, [1, 2, 7, 14, 30])

    def ewm_features(dataframe, alphas, lags):
        dataframe = dataframe.copy()
        for alpha in alphas:
            for lag in lags:
                dataframe['ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                    dataframe['target_value']. \
                        transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
        return dataframe

    alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
    lags = [1, 2, 3, 4, 5, 7, 10]
    ts = ewm_features(ts, alphas, lags)
    ts = pd.get_dummies(ts, columns=['day_of_week', 'month'])
    ts['target_value'] = np.log1p(ts["target_value"].values)

    def smape(preds, target):
        n = len(preds)
        masked_arr = ~((preds == 0) & (target == 0))
        preds, target = preds[masked_arr], target[masked_arr]
        num = np.abs(preds - target)
        denom = np.abs(preds) + np.abs(target)
        smape_val = (200 * np.sum(num / denom)) / n
        return smape_val

    def lgbm_smape(preds, train_data):
        labels = train_data.get_label()
        smape_val = smape(np.expm1(preds), np.expm1(labels))
        return 'SMAPE', smape_val, False

    time = ts.Date
    target = ts.target_value
    data = []
    for i in range(17):
        train, test = train_test_split(ts, train_size=train_size + 7 * i, shuffle=False)

        cols = [col for col in train.columns if col not in ['target_value', 'Date']]
        Y_train = train['target_value']
        X_train = train[cols]
        Y_val = Y_train[-28:]
        X_val = X_train[-28:]
        X_test = test[cols][:7]

        lgbtrain = lgb.Dataset(data=X_train, label=Y_train, feature_name=cols)
        lgbval = lgb.Dataset(data=X_val, label=Y_val, reference=lgbtrain, feature_name=cols)
        lgbtest = lgb.Dataset(data=X_test, feature_name=cols)

        lgb_params = {'metric': {'mae'},
                      'num_leaves': 10,
                      'learning_rate': 0.02,
                      'feature_fraction': 0.8,
                      'max_depth': 5,
                      'verbose': -1,
                      'num_boost_round': 15000,
                      'early_stopping_rounds': 200,
                      'nthread': -1}

        model = lgb.train(lgb_params, lgbtrain,
                          valid_sets=[lgbtrain, lgbval],
                          num_boost_round=lgb_params['num_boost_round'],
                          early_stopping_rounds=lgb_params['early_stopping_rounds'],
                          feval=lgbm_smape,
                          verbose_eval=False)
        y_pred_val = model.predict(X_test)

        y_pred_val = np.expm1(y_pred_val)
        data += list(y_pred_val)

    return data


def my_arima(ts, train_size):
    time = ts.Date
    target = ts.target_value

    data = []
    for i in range(17):
        train_1, test_1 = train_test_split(target, train_size=train_size + 7 * i, shuffle=False)
        model_2 = auto_arima(train_1, start_p=1, start_q=1,
                             test='adf',
                             max_p=5, max_q=5,
                             m=7,
                             d=None,
                             seasonal=True,
                             start_P=0,
                             D=0,
                             trace=False,
                             error_action='ignore',
                             suppress_warnings=True,
                             stepwise=True)
        forecasts_2 = model_2.predict(n_periods=7)
        data = data + list(forecasts_2)

    return data
