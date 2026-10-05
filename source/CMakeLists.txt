# Sketch of the targets, see design.h, section 11.
#
#   libryujin   obj_common
#               obj_<equation> and obj_<equation>_dependent for every
#               enabled hyperbolic description
#   ryujin      main.cc ryujin_equation_dispatch.cc, linked against libryujin
#
# ryujin_add_hyperbolic_description(<name> TYPE <type> HEADER <header>)
# creates obj_<name>_dependent from the four files in dependent/ and
# sets RYUJIN_HYPERBOLIC_DESCRIPTION=<type> and
# RYUJIN_DESCRIPTION_HEADER=<header> on it.
