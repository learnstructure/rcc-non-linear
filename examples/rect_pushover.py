from rcc_non_linear import Model
from rcc_non_linear.utils.helper import plot_response, plot_response_multi
col_props = {
    'fc': 5.5,
    'B': 20, 'H': 30, 'L': 150,
    'cover': 1.5,
    'nBarsTop': 3, 'dbTop': 1,
    'nBarsBot': 5, 'dbBot': 1.27,
    # 'nBarsInt': 6, 'dbInt': 0.984,
    'fy': 68, 'fu': 95, 'Es': 29000, 'e_sh': 0.0115, 'e_ult': 0.12,
    'dh':0.375, 'sh':3, 'fyh':68, 'esm':0.12,
    'nx': 2, 'ny':2,
    'P_axial': 10, #'failure_criteria': ['core', 'strength'], "core_crush_limit": 0.01,
    "divB": 15, "divD": 25
}

model = Model(col_props)

# print(model.confined_props)
# print(model.unconfined_props)
# print(model.fib_section)

#To plot the fiber section, use any of the following methods:
# model.plot_fib_section()
# model.fib_section.plot()

# results_df, bilinear_df, yield_step = model.run_M_phi_analysis()
results_df, bilinear_df, yield_step = model.run_pushover_analysis()
print(model.crushed_cores)
# model.plot_fib_section_damage()
# print(f"Yield occurred at step: {yield_step}")
print(bilinear_df)

plot_response_multi(
    dfs=[results_df.iloc[:, 0:2], bilinear_df],
    names=["Original", "Bilinear"],
    title="Curves Comparison"
)
# plot_response(results_df.iloc[:, 2: 4])
# print("Effective K", model.k_eff)

# model.create_report()
# print("confinement factor is:", model.k_confinement)