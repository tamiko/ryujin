# obj_euler: as before, without equation_dispatch.cc.

add_library(obj_euler OBJECT initial_state_library.cc)

ryujin_add_hyperbolic_description(euler
  TYPE Euler::HyperbolicDescription
  HEADER ryujin/euler/hyperbolic_description.h)
