# Reference profile schema

Reference profiles are generic and method-agnostic. The method label is metadata. Supported energy units are `kJ/mol`, `kcal/mol`, `eV`, and `hartree`. Labels such as `RC`, `TS`, and `PROD` are aliases for `reactant`, `transition_state`, and `product`.

```yaml
reference_profile:
  method_label: user_reference
  energy_unit: kcal/mol
  zero: reactant
  points:
    reactant: {relative_energy: 0.0}
    transition_state: {relative_energy: 18.14}
    product: {relative_energy: -37.35}
  calibration_target:
    profile: [reactant, transition_state, product]
    primary_barrier: [reactant, transition_state]
```
