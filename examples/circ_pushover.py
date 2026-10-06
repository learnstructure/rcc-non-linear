from rcc_non_linear import Model
from rcc_non_linear.utils.helper import plot_response, plot_response_multi
col_props = {
    'fc': 6.1,
    'D': 48, 'L': 324, 'cover': 2,
    'nBars': 18, 'db': 1.41,
    'fy': 75.2, 'fu': 102.4, 'Es': 29000, 'Esh': 1247, 'e_sh': 0.005, 'e_ult': 0.122,
    'dh':0.888, 'sh':6, 'fyh':54.8, 'esm':0.125,
    'P_axial': 570,  #'failure_criteria': ['rebar'], #"core_crush_limit": 0.01, 'rupture_limit': 0.1,
    'nAng': 30, 'nRad':20, 'nRad_cover': 8
}

model = Model(col_props)
# print(model.material.confined_props())

# print(model.confined_props)
# print(model.unconfined_props)
# print(model.fib_section)

#To plot the fiber section, use any of the following methods:
# model.plot_fib_section()
# model.fib_section.plot()

# results_df, bilinear_df, yield_step = model.run_M_phi_analysis()
# print(model.df_m_phi_idealized)
results_df, bilinear_df, yield_step = model.run_pushover_analysis()
# print("Ruptured bar IDs:", model.ruptured_bars)
# model.plot_fib_section_damage()
# print(model.fib_section.fib_sec)
# print("Crushed core IDs:", model.crushed_cores)
# print(model.bar_fibers)
# print(f"Yield occurred at step: {yield_step}")
# print(bilinear_df)
print(results_df)
# print(f"Max Displacement: {max(results_df['displacements']):.2f} in, Peak Lateral Force: {max(results_df['forces']):.2f} kips")
# print("Effective MOI", model.k_eff)
# plot_response_multi(
#     dfs=[results_df.iloc[:, 0:2], bilinear_df],
#     names=["Original", "Bilinear"],
# )
# plot_response(results_df.iloc[:, 2:4])
# print("Effective K", model.k_eff)
# model.create_report()
