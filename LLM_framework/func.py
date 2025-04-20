def calc_rmse(predictions, targets):
    """
    Calculate the Root Mean Square Error (RMSE) between predictions and targets.

    Args:
        predictions (list): List of predicted values.
        targets (list): List of target values.

    Returns:
        float: The RMSE value.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    
    squared_errors = [(pred - target) ** 2 for pred, target in zip(predictions, targets)]
    mean_squared_error = sum(squared_errors) / len(squared_errors)
    rmse = mean_squared_error ** 0.5
    return rmse

def calc_mae(predictions, targets):
    """
    Calculate the Mean Absolute Error (MAE) between predictions and targets.

    Args:
        predictions (list): List of predicted values.
        targets (list): List of target values.

    Returns:
        float: The MAE value.
    """
    if len(predictions) != len(targets):
        raise ValueError("Predictions and targets must have the same length.")
    
    absolute_errors = [abs(pred - target) for pred, target in zip(predictions, targets)]
    mean_absolute_error = sum(absolute_errors) / len(absolute_errors)
    return mean_absolute_error