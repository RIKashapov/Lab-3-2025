from clearml.automation import (
    HyperParameterOptimizer,
    UniformIntegerParameterRange,
    UniformParameterRange,
    DiscreteParameterRange,
)
from clearml.automation.optuna import OptimizerOptuna

BASE_TASK_ID = "4d45675b457f4881836d15ed6d7bd713"

def main():
    optimizer = HyperParameterOptimizer(
        base_task_id=BASE_TASK_ID,

        optimizer_class=OptimizerOptuna,
        max_iteration_per_job=1,

        objective_metric_title="metrics",
        objective_metric_series="RMSE_avg_7d",
        objective_metric_sign="min",

        max_number_of_concurrent_tasks=2,
        execution_queue="default",
        total_max_jobs=10,

        hyper_parameters=[
            UniformIntegerParameterRange("General/model_depth", 4, 10, 1),
            UniformParameterRange("General/model_learning_rate", 0.02, 0.2, 0.01),
            UniformIntegerParameterRange("General/model_iterations", 300, 1200, 100),
            DiscreteParameterRange("General/model_l2_leaf_reg", values=[1, 3, 5, 7, 9]),
        ],
    )

    optimizer.start()
    optimizer.wait()
    optimizer.stop()

if __name__ == "__main__":
    main()
