import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import math
import os

def interpolate_z(df, x, y, method="linear", fallback="nearest"):
    '''Interpolate z for two independent variables x & y'''
    x_data = df.iloc[:, 0].values
    y_data = df.iloc[:, 1].values
    z_data = df.iloc[:, 2].values

    points = np.column_stack((x_data, y_data))

    xq = np.atleast_1d(x)
    yq = np.atleast_1d(y)
    query_points = np.column_stack((xq, yq))

    # Primary interpolation
    z_interp = griddata(points, z_data, query_points, method=method)

    # Fallback for points outside convex hull
    if fallback is not None:
        mask = np.isnan(z_interp)
        if np.any(mask):
            z_interp[mask] = griddata(
                points, z_data, query_points[mask], method=fallback
            )

    return z_interp[0].item() if z_interp.size == 1 else z_interp.item()


def caltrans_bilinear(df, yield_index):
    """
    Generate Caltrans-style bilinear idealization.
    - First branch: elastic slope up to yield, extended linearly until idealized yield
    - Second branch: horizontal (constant strength)
    - Area equality enforced beyond yield point.
    """
    cols = df.columns
    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()

    # Extract key points
    Dy, Vy = x[yield_index], y[yield_index]
    Du, Vu = x[-1], np.max(y)
    Ke = Vy / Dy  # elastic slope

    # Actual area beyond yield
    area_actual = np.trapz(y[yield_index:], x[yield_index:])

    # Function to compute area under bilinear idealized curve (beyond yield)
    def bilinear_area(Dp):
        # beyond yield: from Dy → Dp (still increasing linearly), then horizontal at Vp
        Vp = Vy + Ke * (Dp - Dy)
        area1 = 0.5 * (Vp + Vy) * (Dp - Dy)  # trapezoid under rising branch
        area2 = Vp * (Du - Dp)               # rectangular plateau
        return area1 + area2, Vp

    # Find Dp (plastic hinge start) such that areas match
    Dp_low, Dp_high = Dy, Du
    for _ in range(1000):
        Dp_mid = 0.5 * (Dp_low + Dp_high)
        area_mid, _ = bilinear_area(Dp_mid)
        if abs(area_mid - area_actual) < 1e-3:
            break
        if area_mid > area_actual:
            Dp_high = Dp_mid
        else:
            Dp_low = Dp_mid

    Dp = Dp_mid
    _, Vp = bilinear_area(Dp)
    # Construct bilinear coordinates
    x_bi = [0, Dy, Dp, Du]
    y_bi = [0, Vy, Vp, Vp]
    bilinear_df = pd.DataFrame({cols[0]: x_bi, cols[1]: y_bi})
    return bilinear_df

def load_mander_k():
    return pd.read_csv("rect_conf_k.csv", header=None)


class CircConcreteMander:
    def __init__(self, fc_prime, D, cover, dh, sh, fyh, esm):
        Asp = math.pi * dh**2 / 4
        self.fc_prime = fc_prime
        self.ds = D - 2 * cover - dh
        self.fl_prime = 2 * Asp * fyh / (self.ds * sh)
        self.fyh = fyh
        self.esm = esm
        self.rho = 4 * Asp / (self.ds * sh)
        self.Ec = 57 * math.sqrt(fc_prime * 1000)

    def fcc_prime(self):
        k1 = 2.254 * math.sqrt(1 + 7.94 * self.fl_prime / self.fc_prime)
        return self.fc_prime * (-1.254 + k1 - 2 * self.fl_prime / self.fc_prime)

    def ecc(self):
        return 0.002 * (1 + 5 * (self.fcc_prime() / self.fc_prime - 1))

    def ecu(self):
        return 0.004 + 1.4 * self.rho * self.fyh * self.esm / self.fcc_prime()

    def fc(self, e, fcc_prime, ecc):
        x = e / ecc
        Esec = fcc_prime / ecc
        r = self.Ec / (self.Ec - Esec)
        fc = fcc_prime * x * r / (r - 1 + x**r)
        return fc

    def confined_props(self):
        fcc_prime = self.fcc_prime()
        ecc = self.ecc()
        ecu = self.ecu()
        fc = self.fc(ecu, fcc_prime, ecc)
        return [-fcc_prime, -ecc, -fc, -ecu]

    def unconfined_props(self):
        fc = self.fc(0.005, self.fc_prime, 0.002)
        return [-self.fc_prime, -0.002, -0, -0.005]  # 0 to be replaced by fc

class CircSection:
    def __init__(self, D, cover, Ec, 
                 nBars, db,
                 dh, 
                 sec_tag, core_material, cover_material, bar_material, nAng, nRad, nRad_cover):
        import opsvis as opsv
        self.D, self.cover = D, cover
        self.nBars, self.db = nBars, db
        self.dh = dh
        self.sec_tag = sec_tag
        self.core_tag = core_material
        self.cover_tag = cover_material
        self.bar_tag = bar_material

        self.nAng, self.nRad, self.nRad_cover = nAng, nRad, nRad_cover

        self.R_core = D / 2 - cover - dh / 2
        self.R_bar = D / 2 - cover - dh - db / 2
        self.Ab = math.pi * (db / 2.0) ** 2

        G = Ec / (2 * (1 + 0.2))
        J = 2 * math.pi * (D/2 ** 4) / 4
        GJ = G * J

        self.fib_sec = [
            ['section', 'Fiber', self.sec_tag, '-GJ', GJ],
            ['patch', 'circ', self.core_tag, self.nAng, self.nRad, 0.0, 0.0, 0.0, self.R_core, 0.0, 360.0],
            ['patch', 'circ', self.cover_tag, self.nAng, self.nRad_cover, 0.0, 0.0, self.R_core, self.D/2, 0.0, 360.0],
            ['layer', 'circ', self.bar_tag, self.nBars, self.Ab, 0.0, 0.0, self.R_bar, 180, -180.0 + 360.0 / self.nBars]
        ]
        opsv.fib_sec_list_to_cmds(self.fib_sec)


    def plot(self):
        """
        Plots the fiber section using ops_vis.
        """
        import opsvis as opsv
        opsv.plot_fiber_section(self.fib_sec, fillflag=1, matcolor=['gold', 'lightgrey', 'red'])
        plt.title("Circular Column Fiber Section")
        plt.axis('equal')
        # plt.show()

def run_gravity_analysis(P_axial, type = "MC"):
    import openseespy.opensees as ops
    ops.timeSeries('Constant', 1)
    ops.pattern('Plain', 1, 1)
    if type == "MC":
        ops.load(2, -P_axial, 0.0, 0.0)
    else:
        ops.load(2, 0.0, -P_axial, 0.0)
    ops.integrator('LoadControl', 0.0)
    ops.system('SparseGeneral', '-piv')
    ops.test('NormUnbalance', 1e-6, 50)
    ops.numberer('Plain')
    ops.constraints('Plain')
    ops.algorithm('Newton')
    ops.analysis('Static')
    ops.analyze(1)
    ops.loadConst('-time', 0.0)

def moment_curvature_analysis(model, maxK, dK):
    import openseespy.opensees as ops
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, 0.0)
    
    ops.fix(1, 1, 1, 1)
    ops.fix(2, 0, 1, 0)

    ops.element('zeroLengthSection', 1, 1, 2, model.fib_sec_tag)

    run_gravity_analysis(model.P_axial)

    # Apply moment through node 2 rotation
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(2, 0.0, 0.0, 1.0)
    ops.integrator('DisplacementControl', 2, 3, dK, 1, dK, dK)

    results = {
        'curvatures': [0.0], 'moments': [0.0],
        'Eps_Conc': [0.0], 'Sig_Conc': [0.0],
        'Eps_Steel': [0.0], 'Sig_Steel': [0.0]
    }
    peak_moment = 0.0
    curr_K = 0.0
    yield_curv = None
    step = 0
    while curr_K < maxK:
        ok = ops.analyze(1)
        if ok != 0: break
        step += 1
        ops.reactions()
        curr_moment = -ops.nodeReaction(1, 3)
        curr_K = ops.nodeDisp(2,3)
        if curr_moment > peak_moment: peak_moment = curr_moment
        
        # --- Fiber Responses ---
        # Concrete Core (Material 1) at bottom extreme fiber
        sig_c, eps_c = ops.eleResponse(1, 'section', 'fiber', model.core_h, 0.0, model.core_tag, 'stressStrain')

        # Steel (Material 3) at top extreme fiber
        sig_s, eps_s  = ops.eleResponse(1, 'section', 'fiber', -model.bar_h, 0.0, model.bar_tag, 'stressStrain')

        if (yield_curv is None) and (eps_s >= model.fy / model.Es):
            yield_curv = curr_K
            yield_step = step
            print(f"✅ Yield point reached at curvature = {yield_curv:.6f}")

        results['curvatures'].append(curr_K)
        results['moments'].append(curr_moment)
        results['Eps_Conc'].append(eps_c)
        results['Sig_Conc'].append(sig_c)
        results['Eps_Steel'].append(eps_s)
        results['Sig_Steel'].append(sig_s)

        # Termination Checks
        if curr_moment < 0.85 * peak_moment:
            print(f"⚠️ Strength drop at curvature = {curr_K:.6f}")
            break
        if eps_c < model.confined_props[-1]:
            print(f"⚠️ Concrete crushed at curvature = {curr_K:.6f}")
            break
        if eps_s > model.e_ult:
            print(f"⚠️ Steel ruptured at curvature = {curr_K:.6f}")
            break
        
    results_df = pd.DataFrame(results)
    return results_df, yield_step

def pushover_analysis(model, maxU, dU, self_wt):
    import openseespy.opensees as ops
    ops.node(1, 0.0, 0.0)
    ops.node(2, 0.0, model.L)
    
    ops.fix(1, 1, 1, 1)

    ops.section('Elastic', model.elastic_sec_tag, model.Ec, model.Ag, model.Iz*model.k_eff)

    ops.geomTransf('PDelta', 1)

    ops.beamIntegration('HingeRadau', 1, model.fib_sec_tag, model.lp, model.fib_sec_tag, 0, model.elastic_sec_tag)

    ops.element('forceBeamColumn', 1, *[1, 2], 1, 1)
    
    col_wt = (0.15 / 12**3) * model.Ag * model.L if self_wt else 0.0
    total_wt = model.P_axial + col_wt/2
    print("Total axial load applied on column:", total_wt)
    run_gravity_analysis(total_wt, type="pushover")

    # Apply moment through node 2 rotation
    ops.timeSeries('Linear', 2)
    ops.pattern('Plain', 2, 2)
    ops.load(2, 1.0, 0.0, 0.0)
    ops.integrator('DisplacementControl', 2, 1, -dU)

    results = {
        'displacements': [0.0], 'forces': [0.0],
        'Eps_Conc': [0.0], 'Sig_Conc': [0.0],
        'Eps_Steel': [0.0], 'Sig_Steel': [0.0],
        'drift %': [0.0]
    }
    peak_force = 0.0
    curr_disp = 0.0
    yield_disp = None
    step = 0
    yield_step = None
    while curr_disp < maxU:
        ok = ops.analyze(1)
        if ok != 0: break
        step += 1
        ops.reactions()
        curr_force = ops.nodeReaction(1, 1)
        curr_disp = -ops.nodeDisp(2, 1)
        # print(curr_disp, curr_force)
        if curr_force > peak_force: peak_force = curr_force
        
        # --- Fiber Responses ---
        sig_c, eps_c = ops.eleResponse(1, 'section', model.fib_sec_tag, 'fiber', model.core_h, 0.0, model.core_tag, 'stressStrain')
        sig_s, eps_s  = ops.eleResponse(1, 'section', model.fib_sec_tag, 'fiber', -model.bar_h, 0.0, model.bar_tag, 'stressStrain')

        if (yield_disp is None) and (eps_s >= model.fy / model.Es):
            yield_disp = curr_disp
            yield_step = step
            print(f"✅ Yield point reached at displacement = {yield_disp:.6f}")

        results['displacements'].append(curr_disp)
        results['forces'].append(curr_force)
        results['Eps_Conc'].append(eps_c)
        results['Sig_Conc'].append(sig_c)
        results['Eps_Steel'].append(eps_s)
        results['Sig_Steel'].append(sig_s)
        results['drift %'].append(curr_disp*100/model.L)

        # Termination Checks
        if curr_force < 0.85 * peak_force:
            print(f"⚠️ Strength drop at displacement = {curr_disp:.6f}")
            break
        if eps_c < model.confined_props[-1]:
            print(f"⚠️ Concrete crushed at displacement = {curr_disp:.6f}")
            break
        if eps_s > model.e_ult:
            print(f"⚠️ Steel ruptured at displacement = {curr_disp:.6f}")
            break   
    results_df = pd.DataFrame(results)
    return results_df, yield_step


class Model:
    def __init__(self, props: dict):
        """
        props: dictionary containing column properties
        """
        # Geometry & materials
        self.fc = props["fc"]
        self.D = props.get("D", None)     # diameter for circular section
        self.B = props.get("B", None)    # width for rectangular section  
        self.H = props.get("H", None)   #depth for rectangular section
        self.L = props["L"]         # length of the column

        self.section_type = "rectangular" if self.B and self.H else "circular"

        self.cover = props["cover"]      # concrete cover   

        if self.section_type == "circular":
            self.nBars = props["nBars"]
            self.db = props["db"]
            self.nAng = props.get("nAng", 30)
            self.nRad = props.get("nRad", 20)
            self.nRad_cover= props.get("nRad_cover", 8)
        elif self.section_type == "rectangular":
            self.nBarsTop = props["nBarsTop"]
            self.dbTop = props["dbTop"]
            self.nBarsBot = props["nBarsBot"]
            self.dbBot = props["dbBot"]
            self.nBarsInt = props.get("nBarsInt", 0)
            self.dbInt = props.get("dbInt", 0)
            self.nx = props.get("nx", 2)   # number of transverse bars in x direction
            self.ny = props.get("ny", 2)   # number of transverse bars in y direction
            self.divB = props.get("divB", 30)
            self.divD = props.get("divD", 30)
            self.divCover = props.get("divCover", 5)
          
        else:
            raise ValueError("Either B and H (rectangular) or D (circular) must be provided.")

        self.fy = props["fy"]
        self.fu = props["fu"]
        self.Es = props.get("Es", 29000)   # default modulus of elasticity
        self.Esh = props.get("Esh", 0.043 * self.Es)   # default strain hardening modulus
        self.e_sh = props.get("e_sh", 0.005)
        self.e_ult = props.get("e_ult", 0.1)

        self.dh = props["dh"]
        self.sh = props["sh"]
        self.fyh = props.get("fyh", 68)   # default transverse reinforcement yield strength
        self.fuh = props.get("fuh", 95)   # default transverse reinforcement ultimate strength
        self.esm = props.get("esm", 0.1)  # default transverse reinforcement ultimate strain
        self.P_axial = props.get("P_axial", 0.0)  # axial load

        self.core_tag, self.cover_tag, self.bar_tag = 1, 2, 3  # material tags
        self.fib_sec_tag, self.elastic_sec_tag = 1, 2

        # Derived
        self.Ec = 57 * math.sqrt(self.fc * 1000)

        if self.section_type == "circular":
            self.Ag = math.pi * (self.D**2) / 4
            self.As = self.nBars * math.pi * (self.db**2) / 4
            self.Iz = math.pi * ((self.D/2) ** 4) / 4
        else:
            self.Ag = self.B * self.H
            self.As = (self.nBarsTop + self.nBarsBot + self.nBarsInt) * math.pi * (self.dbTop**2) / 4
            self.Iz = (self.B * self.H**3) / 12.0

        db = self.db if self.section_type == "circular" else max(self.dbTop, self.dbBot)
        self.lp = max(0.08 * self.L + 0.15 * self.fy * db, 0.3*db*self.fy)
        self.m_phi_done = False
        self.create_model()        

    def create_model(self):
        import openseespy.opensees as ops
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        self.define_materials()
        self.define_section()

    def define_materials(self):
        import openseespy.opensees as ops
        if self.section_type == "circular":
            material = CircConcreteMander(
                fc_prime=self.fc, D=self.D, cover=self.cover,
                dh=self.dh, sh=self.sh,
                fyh=self.fyh, esm=self.esm
            )   
        else:
            material = RectConcreteMander(
                fc_prime=self.fc,
                B=self.B, H=self.H, cover=self.cover,
                dh=self.dh, sh=self.sh,
                fyh=self.fyh, esm=self.esm,
                nx=self.nx, ny=self.ny
            )
            self.k_confinement = material.k
        self.confined_props = material.confined_props()
        self.unconfined_props = material.unconfined_props()
        ops.uniaxialMaterial('Concrete01', self.core_tag, *self.confined_props)
        ops.uniaxialMaterial('Concrete01', self.cover_tag, *self.unconfined_props)
        ops.uniaxialMaterial('ReinforcingSteel', self.bar_tag, self.fy, self.fu, self.Es, self.Esh, self.e_sh, self.e_ult)
    
    def define_section(self):
        if self.section_type == "circular":
            self.fib_section = CircSection(self.D, self.cover, self.Ec,
                                           self.nBars, self.db, self.dh, 
                                           self.fib_sec_tag, self.core_tag, self.cover_tag, self.bar_tag, self.nAng, self.nRad, self.nRad_cover)
            self.core_h = self.fib_section.R_core
            self.bar_h = self.fib_section.R_bar
        else:
            self.fib_section = RectSection(
                B=self.B, H=self.H, cover=self.cover, Ec=self.Ec,
                nBarsTop=self.nBarsTop, dbTop=self.dbTop,
                nBarsBot=self.nBarsBot, dbBot=self.dbBot,
                nBarsInt=self.nBarsInt, dbInt=self.dbInt,
                dh=self.dh,
                sec_tag=self.fib_sec_tag, core_material=self.core_tag,
                cover_material=self.cover_tag, bar_material=self.bar_tag,
                divB = self.divB, divD = self.divD, divCover = self.divCover
            )
            self.core_h = self.fib_section.core_h
            self.bar_h = self.fib_section.bar_h
        

    def plot_fib_section(self, save_path=None):
        self.fib_section.plot()       # draw the figure
        import matplotlib.pyplot as plt
        if save_path:                 # if a path is provided
            plt.savefig(save_path, bbox_inches='tight')  # save current figure
            plt.close()               # close figure to free memory
        else:
            plt.show()                # just display interactively

    def run_M_phi_analysis(self, maxK=0.02, dK=0.00002):
        if maxK is None: maxK = 50 * (self.fy / self.Es) / self.core_h
        if dK is None: dK = maxK / 1000 
        self.create_model()  
        results_df, yield_step = moment_curvature_analysis(self, maxK, dK)
        bilinear_df = caltrans_bilinear(results_df, yield_step)
        self.k_eff = self.get_stiffness_modifier(bilinear_df)
        self.m_phi_done = True
        self.df_m_phi, self.df_m_phi_idealized = results_df, bilinear_df
        return results_df, bilinear_df, yield_step

    def get_stiffness_modifier(self, bilinear_df):
        phiY, mY = bilinear_df.iloc[1, 0], bilinear_df.iloc[1, 1]
        I_eff = mY/(phiY*self.Ec)
        return I_eff/ self.Iz

    def run_pushover_analysis(self, maxU=None, dU=0.05, self_wt=True):
        if maxU is None: maxU = 0.2 * self.L
        if not self.m_phi_done:
            self.run_M_phi_analysis()
        self.create_model()  
        results_df, yield_step = pushover_analysis(self, maxU, dU, self_wt)
        bilinear_df = caltrans_bilinear(results_df, yield_step)
        bilinear_df["drift %"] = bilinear_df["displacements"] * 100 / self.L
        self.df_pushover, self.df_pushover_idealized = results_df, bilinear_df
        return results_df, bilinear_df, yield_step


def plot_response(
    df,
    x_label=None,
    y_label=None,
    title="Response Curve",
    grid=True,
    show=True,
    figsize=(8, 5),
):
    """Plots a 2D response curve using matplotlib."""
    import matplotlib.pyplot as plt

    x = df.iloc[:, 0]
    y = df.iloc[:, 1]

    x_label = x_label or str(df.columns[0])
    y_label = y_label or str(df.columns[1])

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x, y, linewidth=2, color="#1f77b4")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(y_label, fontsize=11)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


def plot_response_multi(
    dfs,
    names=None,
    colors=None,
    x_label=None,
    y_label=None,
    title="Response Curve",
    grid=True,
    show=True,
    figsize=(8, 5),
):
    """Plots multiple response curves on the same matplotlib axes for comparison."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    for i, df in enumerate(dfs):
        x = df.iloc[:, 0]
        y = df.iloc[:, 1]

        label = names[i] if names and i < len(names) else f"Trace {i+1}"
        color = colors[i] if colors and i < len(colors) else None

        ax.plot(x, y, label=label, linewidth=2, color=color)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel(x_label or str(dfs[0].columns[0]), fontsize=11)
    ax.set_ylabel(y_label or str(dfs[0].columns[1]), fontsize=11)
    ax.legend(frameon=True)
    if grid:
        ax.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    if show:
        plt.show()

    return fig, ax


if __name__ == "__main__":
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

    # results_df, bilinear_df, yield_step = model.run_M_phi_analysis()
    results_df, bilinear_df, yield_step = model.run_pushover_analysis()

    # print(f"Yield occurred at step: {yield_step}")
    print(results_df)

    # plot_response_multi(
    #     dfs=[results_df.iloc[:, 0:2], bilinear_df],
    #     names=["Original", "Bilinear"],
    # )
