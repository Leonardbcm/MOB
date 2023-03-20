import os, pandas, copy, ast, numpy as np
from ast import literal_eval as make_tuple
from src.models.torch_models.weight_initializers import initializers_from_string
import src.models.spliter as SPL

def base_folder():
    return os.environ['MOB']

def flatten_dict(d):
    res = []
    for k in d.keys():
        if type(d[k]) is list:
            res += d[k]
        else:
            res += d[k].tolist()
            
    return res

def default_dataset_path(dataset_name):
    return os.path.join(base_folder(), "data", "datasets", dataset_name)

def all_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "all.csv")

def extra_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "extra.csv")

def train_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "train.csv")

def val_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "val.csv")

def test_dataset_path(dataset_name):
    return os.path.join(default_dataset_path(dataset_name), "test.csv")

def folder(dataset_name):
    return os.path.join(base_folder(), "data", "datasets", dataset_name)

def figure_folder(dataset_name):
    return os.path.join(base_folder(), "figures", dataset_name)

def save_name(prefix, dataset_name):
    return prefix + "_" + dataset_name

def column_path(dataset_name, order_str):
    return os.path.join(folder(dataset_name), f"columns_{order_str}.txt")

def load_columns(dataset_name, order_str=""):
    with open(column_path(dataset_name, order_str), "r") as f:
        columns = f.read()
    columns = columns.split('", "')
    columns[0] = columns[0][2:]
    columns[-1] = columns[-1][:-3]
    if columns == ["y"]:
        columns = []
    return columns

def load_labels(dataset_name):
    with open(os.path.join(folder(dataset_name), "labels.txt"), "r") as f:
        columns = f.read()
    columns = columns.split('", "')
    columns[0] = columns[0][2:]
    columns[-1] = columns[-1][:-3]
    return columns    

def save_path(prefix, dataset_name):
    return os.path.join(folder(dataset_name), save_name(prefix, dataset_name))

def model_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_model"

def train_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_train_predictions.csv"

def val_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_val_predictions.csv"

def test_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_test_predictions.csv"

def test_recalibrated_prediction_path(prefix, dataset_name, replace_ATC, filters,
                                      inverted_filters):
    if replace_ATC != "":
        replace_ATC = "_" + replace_ATC
        
    s_filters = ""
    if filters is not None:
        for col in filters:
            s_filters += f"_{col}={filters[col]}"
    if inverted_filters is not None:
        for col in inverted_filters:
            s_filters += f"_{col}!={inverted_filters[col]}"            
        
    return save_path(prefix + replace_ATC + s_filters, dataset_name) + "_test_recalibrated_predictions.csv"

def test_shape_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_test_shape_values.npy"

def test_recalibrated_shape_path(prefix, dataset_name, replace_ATC):
    if replace_ATC != "":
        replace_ATC = "_" + replace_ATC
    return save_path(prefix + replace_ATC, dataset_name) + "_test_recalibrated_shape_values.npy"

def all_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_all_predictions.csv"

def extra_prediction_path(prefix, dataset_name):
    return save_path(prefix, dataset_name) + "_extra_predictions.csv"

def mae_epsilon_(ytrue, ypred, epsilon=0.01):
    if abs(ytrue - ypred) < epsilon: return 0.0
    return abs(ytrue - ypred)

def mae_epsilon(ytrue, ypred, epsilon=0.01):
    s = 0
    for yt, yp in zip(ytrue, ypred):
        s += mae_epsilon_(yt, yp, epsilon=epsilon)

    return s/len(ytrue)

def smape_epsilon_(ytrue, ypred, epsilon=0.01):
    if abs(ytrue - ypred) < epsilon: return 0.0
    return 2 * abs(ytrue - ypred) / (abs(ytrue) + abs(ypred))

def smape_epsilon(ytrue, ypred, epsilon=0.01):
    s = 0
    for yt, yp in zip(ytrue, ypred):
        s += smape_epsilon_(yt, yp, epsilon=epsilon)

    return 100 * s/len(ytrue)

def gain_(yt, y, t, sigma=0, epsilon=0.01, volume=25, epsilon_percentage=False):
    if not epsilon_percentage:
        if y - t > epsilon: return 24 * volume * (yt - t - sigma)
        if t - y > epsilon: return 24 * volume * (t - yt - sigma)
        return 0.0
    else:
        if y > (1 + epsilon) * t: return 24 * volume * (yt - t - sigma)
        if y < (1 - epsilon) * t: return 24 * volume * (t - yt - sigma)
        return 0.0

def gain(ytrue, ypred, trades, sigma=0, epsilon=0.01, volume=25,
         epsilon_percentage=False):
    s = 0
    for yt, yp, t in zip(ytrue, ypred, trades):
        s += gain_(yt, yp, t, sigma=sigma, epsilon=epsilon, volume=volume,
                   epsilon_percentage=epsilon_percentage)

    return s

def direction_(yt, yp, t):
    return (yt > t and yp > t) or (yt < t and yp < t)

def direction(ytrue, ypred, trades):
    s = 0
    for yt, yp, t in zip(ytrue, ypred, trades):
        s += direction_(yt, yp, t)

    return 100 * s/len(ytrue)

def past_label_columns(dataset, past_label):
    try:
        past_labels = [np.where(dataset.columns == c)[0][0] for c in past_label]
    except:
        past_labels = None
    
    return past_labels

def load_dataset(path, label, past_label):
    dataset = pandas.read_csv(path)
    labels = dataset[label]
    dataset.drop(columns=np.concatenate((np.array(label),
                                         np.array(["period_start_date"]))),
                 inplace=True)
    past_label_idx = past_label_columns(dataset, past_label)
    return dataset, labels, past_label_idx

def load_dataset_(dataset, labels, label):
    X = dataset.values
    y = labels.values
    if len(label) == 1:
        y = y.reshape(1, -1).ravel()
    return X, y

def neurons_per_layer_to_string(npl):
    return str(npl)

def neurons_per_layer_from_string(res):
    return make_tuple(res)

def spliter_to_string(spliter):
    return str(spliter)

def spliter_from_string(res):
    clas, val, shuffle = res.split("_")
    obj = getattr(SPL, clas)
    val = float(val)
    shuffle = bool(shuffle)
    return obj(val, shuffle=shuffle)

def filter_size_to_string(fs):
    return str(fs)

def filter_size_from_string(res):    
    return make_tuple(res)

def dilation_rate_to_string(dr):
    return str(dr)

def dilation_rate_from_string(res):
    return make_tuple(res)

def weight_initializer_to_string(wi):
    return [str(w) for w in wi]

def weight_initializer_from_string(res):
    return initializers_from_string(res)

def activity_regularizer_to_string(activity_regularizer):
    if activity_regularizer is not None:
        res = activity_regularizer.get_config()
    else: res = ""
    return res

def activity_regularizer_from_string(res):
    if res == "": return  None    
    if pandas.isna(res): return None

    res = ast.literal_eval(res)
    typ = [k for k in res.keys()]
    if len(typ) == 1:
        typ = typ[0]
    else:
        typ = "L1L2"

    return getattr(regularizers, typ)(**res)

def scale_mae(mae):
    return -mae

def to_csv(grid_search, results_path, map_dict={}):
    n_iter = len(grid_search.cv_results_['params'])
    params = [k for k in grid_search.cv_results_['params'][1].keys()]
    splits = [k for k in grid_search.cv_results_.keys() if 'split' in k]
    measure_col = "measure_type"
    result_col = "measure_value"
    config_col = "config"

    columns = params + [measure_col, result_col, config_col]
    for_each_conf = splits + ['mean_test_score', 'std_test_score', 'rank_test_score',
                              'mean_fit_time', 'std_fit_time']

    df = pandas.DataFrame(columns=columns)
    for i in range(n_iter):
        config = grid_search.cv_results_['params'][i]        
        dftemp = pandas.DataFrame(columns=columns)
        for j, row in enumerate(for_each_conf):
            dftemp.loc[j, config_col] = i
            dftemp.loc[j, measure_col] = row

            base_metric = grid_search.cv_results_[row][i]
            value = base_metric if row not in splits + ["mean_test_score"] else scale_mae(base_metric)
            dftemp.loc[j, result_col] = value
            for param in params:
                dftemp.loc[j, param] = config[param]
        df = df.append(dftemp)

    # Save all the results
    df.to_csv(results_path[:-4] + "raw.csv", index=False)

    # Save formatted results
    df_part = df.loc[df.measure_type == "mean_test_score", params + ["measure_value"]]
    df_part_time = df.loc[df.measure_type == "mean_fit_time", params + ["measure_value"]]
    df_part.rename(columns={"measure_value" : "maes"}, inplace=True)
    df_part_time.rename(columns={"measure_value" : "times"}, inplace=True)    
    df_part_time.times *= len(splits)

    df_part.index = [i for i in range(df_part.shape[0])]
    df_part_time.index = [i for i in range(df_part_time.shape[0])]    
    df_part["times"] = df_part_time.times

    new_names = [k.split("__")[-1] for k in df_part.drop(columns=["maes", "times"]).keys()]
    name_mapper = {}
    for (old_name, new_name) in zip(df_part.drop(columns=["maes", "times"]).keys(), new_names):
        name_mapper[old_name] = new_name
    df_part.rename(columns=name_mapper, inplace=True)

    # Convert objects into string
    for k in map_dict.keys():
        v = map_dict[k]
        df_part[k] = [v[0](i) for i in df_part[k]]
        
    print("Best configuration is : ")
    print(df_part.loc[df_part.maes.values.argmin()])
        
    df_part.to_csv(results_path, index=False)    
    return df_part

def load_results(path, map_dict):
    df = pandas.read_csv(path)
    for k in map_dict.keys():
        # Handle nested params
        if type(map_dict[k]) == dict:
            for k2 in map_dict[k].keys():
                _, f = map_dict[k][k2]
                df[k2] = [f(i) for i in df[k2]]
        else:
            _, f = map_dict[k]
            if k in df.keys(): df[k] = [f(i) for i in df[k]]
    return df        
