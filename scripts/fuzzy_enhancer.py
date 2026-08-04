import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define fuzzy input variables
thickness = ctrl.Antecedent(np.arange(0, 15.1, 0.1), 'thickness')
curvature = ctrl.Antecedent(np.arange(0, 50.1, 1), 'curvature')   # average contour complexity

# Define fuzzy output variable
enhancement = ctrl.Consequent(np.arange(0, 101, 1), 'enhancement')  # enhancement strength %

# Membership functions for thickness
thickness['thin'] = fuzz.trimf(thickness.universe, [0, 1, 3])
thickness['medium'] = fuzz.trimf(thickness.universe, [2, 5, 8])
thickness['thick'] = fuzz.trimf(thickness.universe, [6, 10, 15])

# Membership functions for curvature
curvature['low'] = fuzz.trimf(curvature.universe, [0, 10, 20])
curvature['moderate'] = fuzz.trimf(curvature.universe, [15, 25, 35])
curvature['high'] = fuzz.trimf(curvature.universe, [30, 40, 50])

# Membership functions for enhancement
enhancement['low'] = fuzz.trimf(enhancement.universe, [0, 20, 40])
enhancement['medium'] = fuzz.trimf(enhancement.universe, [30, 50, 70])
enhancement['high'] = fuzz.trimf(enhancement.universe, [60, 80, 100])

# Fuzzy rules
rule1 = ctrl.Rule(thickness['thin'] & curvature['low'], enhancement['high'])
rule2 = ctrl.Rule(thickness['thin'] & curvature['moderate'], enhancement['high'])
rule3 = ctrl.Rule(thickness['thin'] & curvature['high'], enhancement['medium'])

rule4 = ctrl.Rule(thickness['medium'] & curvature['low'], enhancement['medium'])
rule5 = ctrl.Rule(thickness['medium'] & curvature['moderate'], enhancement['medium'])
rule6 = ctrl.Rule(thickness['medium'] & curvature['high'], enhancement['low'])

rule7 = ctrl.Rule(thickness['thick'] & curvature['low'], enhancement['low'])
rule8 = ctrl.Rule(thickness['thick'] & curvature['moderate'], enhancement['low'])
rule9 = ctrl.Rule(thickness['thick'] & curvature['high'], enhancement['low'])

# Build control system
enhancement_ctrl = ctrl.ControlSystem([
    rule1, rule2, rule3,
    rule4, rule5, rule6,
    rule7, rule8, rule9
])
enhancer = ctrl.ControlSystemSimulation(enhancement_ctrl)

# Interface function
def get_enhancement_strength(input_thickness, input_curvature):
    try:
        enhancer.input['thickness'] = input_thickness
        enhancer.input['curvature'] = input_curvature
        enhancer.compute()
        return enhancer.output['enhancement']
    except Exception as e:
        print(f"[ERROR] Fuzzy logic computation failed: {e}")
        return 0.0  # fallback to no enhancement
