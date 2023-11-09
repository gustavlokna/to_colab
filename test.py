def hyperparameter_tuning(param_grid, data, est):

    # Create the scorer
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)

    # Create the grid search
    grid_search = GridSearchCV(
        estimator=est, param_grid=param_grid, scoring=scorer,
        cv = data.split_index() 
    )

    # Fit the grid search
    grid_search.fit(data.X, data.y["pv_measurement"])

    # Get the best parameters
    best_params = grid_search.best_params_
    print(f"Best parameters b: {best_params}")
    return best_params