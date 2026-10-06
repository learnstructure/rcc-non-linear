from rcc_non_linear import Model
from rcc_non_linear.utils.helper import plot_response, plot_response_multi

col_props = {
    'fc': 5,
    'B': 36, 'H': 12, 'L': 132,
    'cover': 2,
    'nBarsTop': 6, 'dbTop': 0.875,
    'nBarsBot': 6, 'dbBot': 0.875,
    'nBarsInt': 0, 'dbInt': 0.9843,
    'fy': 68, 'fu': 95, 'Es': 29000,
    'dh':0.625, 'sh':12, 'fyh':68, 'esm':0.1,
    'nx': 2, 'ny':0,
    'P_axial': 3.3,
}

model = Model(col_props)

# print(model.confined_props)
# print(model.unconfined_props)
# print(model.ke)

#To plot the fiber section, use any of the following methods:
model.plot_fib_section()
# model.fib_section.plot()
#To run moment-curvature analysis:
results_df, bilinear_df, yield_step = model.run_M_phi_analysis()
# print(f"Yield occurred at step: {yield_step}")
print(bilinear_df)

plot_response_multi(
    dfs=[results_df.iloc[:, 0:2], bilinear_df],
    names=["Original", "Bilinear"],
)
# print("Effective K", model.k_eff)
