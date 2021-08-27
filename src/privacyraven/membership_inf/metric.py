"""
import numbers
from privacyraven.extraction.synthesis import process_data
#metric_functions = ["aucroc"]

metrics = dict()

def register_metric(func):
    metrics[func.__name__] = func
    return func

def calculate_metric_value(metric, data_point, query_substitute,
                           substitute_model, extraction_attack, loss=None,
                           threshold=None):
    # Threshold must be: 
    # - a number 
    # - a string in a list of function strings
    # - a callable 

    is_number = is_instance(metric, numbers.Number)
    is_proper_string = metric in metric_functions
    is_callable = callable(metric)

    if (is_number || is_proper_string || is_callable) is False:
        raise ValueError("Metric must be a number, a string representing the
                         name of a value metric function, or a callable that
                         can calculate the value of the metric")


@register_metric
def prediction_correctness(data_point, query_substitute,
                           substitute_model, extraction_attack, loss=None,
                           threshold=None):
    # There must be a correct answer to the data point attached 
    (x_data, y_data) = process_data(data_point)
    prediction = query_substitute(x_data)
    print(prediction)
    print(y_data)
    if (prediction == y_data):
        return "This data point is likely a member of the training dataset."
    else:
        return "This datapoint is not likely to be a member of the training
              dataset."

@register_metric
def prediction_loss(data_point, query_substitute,
                           substitute_model, extraction_attack, loss=None,
                           threshold=None):
    if loss(data_point) > threshold:
        return "This data point is not likely to be a member of the training
    dataset."
    else:
        return "This data point is likely a member of the training dataset."


@register_metric
def prediction_confidence(data_point, query_substitute,
                           substitute_model, extraction_attack, loss=None,
                           threshold=None):
"""
