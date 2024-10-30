# %% Test Code
from metrics.metrics import ts_metrics, point_adjustment
from model.params import get_model_size

def tester(best_clf, X_test, y_test):
    best_metric_dict = best_clf.decision_function(X_test, y_test)
    # Confusion Matrix Plot

    # eval_metrics = ts_metrics(y_test, scores)
    # adj_eval_metrics = ts_metrics(y_test, point_adjustment(y_test, scores))

    print("\033[48;5;250m" + f"\033[30m[Best Kernel Number] : {best_clf.num_kernels}\033[0m")

    print("\033[48;5;250m" + f"\033[30m[Best Metric] : {best_metric_dict}\033[0m")

    get_model_size(best_clf)
    
    return best_clf, best_metric_dict