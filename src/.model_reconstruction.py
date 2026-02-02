import pysindy as ps
from utils.custom_libraries import FixedCustomLibrary, FixedWeakPDELibrary
from utils.custom_libraries import x, cos_x, x_cos_y, name_x, name_cos_x, name_x_cos_y
from utils.helpers import compute_time_vector
from data_processing.data_loader import load_data

X, U, time_step = load_data()
x_val, u_val, _ = load_data()


feature_library = FixedWeakPDELibrary(
    function_library=FixedCustomLibrary([x, cos_x, x_cos_y], [x], include_bias=False),
    H_xt=[0.175],
    K=50,
    p=3,
    derivative_order=1,
    differentiation_method=ps.FiniteDifference()
    spatiotemporal_grid=compute_time_vector()
)
optimizer = ps.EnsembleOptimizer(
    opt=ps.STLSQ(alpha=0.01, max_iter=100000, normalize_columns=True, threshold=0.67564185),
    bagging=True, n_models=50, n_subset=1500,
)

ps.