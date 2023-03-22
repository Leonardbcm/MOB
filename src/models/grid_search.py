import pandas, copy, time

from src.models.spliter import MySpliter
import src.models.parallel_scikit as ps

############################# Utility function for model wrappers
def create_mw(country, dataset, model, name, model_params={}, spliter=None,
              EPF="EPF"):
    dataset_ = f"{EPF}{dataset}"
    if EPF == "EPF": dataset_ += "_" + str(country)
    name_ = model.string(model) + "_" + name
    return model(name_, dataset_, country=country, spliter=spliter, **model_params)

def get_search_space(model_wrapper, n, stop_after=-1,
                     dataset="", fast=False, load_from=None):
    if load_from is None:
        return model_wrapper.get_search_space(n=n, fast=fast, stop_after=stop_after)

    # Load results from the original version
    base_model_wrapper = copy.deepcopy(model_wrapper)
    base_model_wrapper.dataset_name = base_model_wrapper.dataset_name.replace(
        dataset, load_from)
    return base_model_wrapper.load_results()

###################### DEFAULT RUN PARAMS
def default_params():
    return {
        # TASKS
        "GRID_SEARCH" : False,
        "LOAD_AND_TEST" : False,
        "RECALIBRATE" : False,   

        # GENERAL PARAMS
        "name" : "TSCHORA",
        "datasets" : ("Lyon",),
        "base_dataset_names" : ("Lyon",),
        "EPF" : "",
        "for_graph" : False,
        "n_val" : 365,
        "n_cpus" : 1,
        "filters" : {},
        "inverted_filters" : {},
        "models" : ([], []),
        "countries" : ("FR", ),
    
        # GRID SEARCH PARAMS
        "fast" : False,
        "n_combis" : 1,
        "restart" : False,
        "n_rep" : 1,
        "stop_after" : -1,
        "save_preds" : False,
        
        # RECALIBRATION PARAMS
        "acc" : False,
        "recompute" : False,    
        "n_shap" : 0,
        "start" : 0,
        "step" : 30,
        "stop" : 731,        
        "calibration_window" : 1440,
    }

############################# MAIN functions
def run(**kwargs):
    
    ######## PARSE PARAMETERS
    params = default_params()
    params.update(kwargs)
    kwargs = params
    
    ## TASKS
    GRID_SEARCH = kwargs["GRID_SEARCH"]
    RECALIBRATE = kwargs["RECALIBRATE"]

    ## GENERAL PARAMS
    GLOBAL_SEED = kwargs["GLOBAL_SEED"]
    name = kwargs["name"]
    datasets = kwargs["datasets"]
    EPF = kwargs["EPF"]
    for_graph = kwargs["for_graph"]
    base_dataset_names = kwargs["base_dataset_names"]    
    n_val = kwargs["n_val"]    
    models = kwargs["models"]
    countries = kwargs["countries"]

    ## GRID SEARCH PARAMS    
    fast = kwargs["fast"]
    n_combis = kwargs["n_combis"]
    restart = kwargs["restart"]
    n_rep = kwargs["n_rep"]    
    stop_after = kwargs["stop_after"]
    n_cpus = kwargs["n_cpus"]
    save_preds = kwargs["save_preds"]

    ## RECALIBRATION PARAMS
    acc = kwargs["acc"]    
    n_shap = kwargs["n_shap"]
    start_rec = kwargs["start"]
    step = kwargs["step"]
    stop_rec = kwargs["stop"]
    calibration_window= kwargs["calibration_window"]
    filters = kwargs["filters"]
    inverted_filters = kwargs["inverted_filters"]
    recompute = kwargs["recompute"]     
    
    ## Walk through 1) datasets 2) countries 3) models 4) repetitions
    recalibration_times = pandas.DataFrame(columns=["country", "model", "times"])
    restart_changed = False
    for (dataset, base_dataset_name) in zip(datasets, base_dataset_names):
        for country in countries:
            for model in models:
                # Get per-models parameters if specified
                if type(model) is list:
                    model_params = model[1]
                    model = model[0]
                else:
                    model_params = {}
                
                if "n_cpus" in model_params:
                    n_cpus_ = model_params.pop("n_cpus")
                else:
                    n_cpus_ = n_cpus

                if "n_combis" in model_params:
                    n_combis_ = model_params.pop("n_combis")
                else:
                    n_combis_ = n_combis
                    
                start = time.time()
                if restart_changed:
                    restart_changed = False
                    restart = True
                
                if GRID_SEARCH:
                    # Performs n_rep repetition of the grid search. A repetition is
                    # evaluating a certain number of combination, storing the
                    # results and disaplaying the best obtained metric and elapsed
                    # time
                    for i in range(n_rep):
                        model_wrapper = run_grid_search(
                            name, dataset, model, country, base_dataset_name,
                            fast, EPF, n_combis_, restart, stop_after, n_cpus_,
                            n_val, model_params, i, save_preds, GLOBAL_SEED)

                        if (n_rep > 1) and restart:
                            print(
                                "Step specified but wram start is not allowed.")
                            print("Disabling restart.")
                            restart_changed = True
                            restart = False
                    
                        df = model_wrapper.load_results()
                        best_params = model_wrapper.best_params(df)
                        print(f"LOSS = {best_params['maes']}")
                        print(f"TIME = {round((time.time() - start)  / 3600, ndigits=2)}h")
                    
                if RECALIBRATE:
                    pause = time.time()
                    total_time = run_recalibrate(
                        name, dataset, model, country, n_cpus_,
                        start_rec, stop_rec, step, n_shap, n_val,
                        base_dataset_name, calibration_window, model_params, EPF,
                        for_graph, filters, inverted_filters, recompute, acc)
                    recalibration_times = pandas.concat(
                        (recalibration_times,
                         pandas.DataFrame(
                             index=[0],
                             columns = ["country", "times", "models"],
                             data = [[country, total_time, model.string(model)]])),
                        ignore_index=True)
                    start = start - (time.time() - pause)

def run_grid_search(name, dataset, model, country, base_dataset_name, fast, EPF,
                    n_combis, restart, stop_after, n_cpus, n_val, model_params,
                    i, save_preds, GLOBAL_SEED):
    # During grid search, the validation set is the last year of data
    spliter = MySpliter(n_val, shuffle = False)

    # Instantiate the Model Wrapper
    model_wrapper = create_mw(country, dataset, model, name, EPF=EPF,
                              model_params=model_params, spliter=spliter)
    
    X, y = model_wrapper.load_train_dataset()
    n = X.shape[0]
    if base_dataset_name == dataset: load_from = None
    else: load_from = base_dataset_name
    search_space = get_search_space(
        model_wrapper, n, dataset=dataset, fast=fast,
        load_from=load_from, stop_after=stop_after)

    print("STARTING REPETITION : ", str(i))
    
    # This makes sure that all the models will have the same sampled parameters
    ps.set_all_seeds(GLOBAL_SEED)
    param_list, seeds = ps.get_param_list_and_seeds(
        search_space, n_combis, model_wrapper=model_wrapper, restart=restart)
    results = ps.parallelize(n_cpus, model_wrapper, param_list, X, y,
                             seeds=seeds, restart=restart, save_preds=save_preds)
    df = ps.results_to_df(results, param_list, seeds=seeds, n_cpus=n_cpus,
                          map_dict=model_wrapper.map_dict(), cv=1)
    if not restart:
        # Dont use load results here because it will parse string as python objects!
        df = pandas.concat((pandas.read_csv(model_wrapper.results_path()), df),
                           ignore_index=True)
    df.to_csv(model_wrapper.results_path(), index=False)
    return model_wrapper

def run_recalibrate(name, dataset, model, country, n_cpus, start, stop, step,
                    n_shap, n_val, base_dataset_name, calibration_window,
                    model_params,EPF, for_graph, filters, inverted_filters,
                    recompute, acc):
    spliter = MySpliter(n_val, shuffle = True)
    model_wrapper = create_mw(country, dataset, model, name, EPF=EPF,
                              model_params=model_params, spliter=spliter)

    base_model_wrapper = copy.deepcopy(model_wrapper)
    base_model_wrapper.dataset_name = base_model_wrapper.dataset_name.replace(
        dataset, base_dataset_name)

    # This is for the Graph-Based models
    base_model_wrapper.load_train_dataset()
    model_wrapper.load_train_dataset()
    
    # Load the best params from this mw
    print(f"LOADING THE BEST CONFIGURATION FOR {model_wrapper.string()} FROM DATASET '{base_dataset_name}' WITH ATCs REPLACED BY '{model_wrapper.replace_ATC}'.")
    for f in filters:
        print(f"KEEPING ONLY THE VALUES {f}={filters[f]}\n")
    for f in inverted_filters:
        print(f"KEEPING ONLY THE VALUES {f}!={inverted_filters[f]}\n")        
    df = base_model_wrapper.load_results()
    best_params = base_model_wrapper.best_params(
        df, for_recalibration=True, acc=acc, recompute=recompute,
        filters=filters, inverted_filters=inverted_filters)
    
    # Recalibrate
    total_time = model_wrapper.recalibrate_epf(
        seed=best_params["seeds"], ncpus=n_cpus,
        calibration_window=calibration_window, filters=filters,
        inverted_filters=inverted_filters,
        best_params=best_params, n_shap=n_shap, start=start, stop=stop, step=step)
    return total_time

