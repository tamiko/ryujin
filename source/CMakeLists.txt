# Sketch of the targets, see design.h, section 11.
#
#   libryujin   obj_common
#               obj_<equation> and obj_<equation>_typed for every
#               enabled hyperbolic description
#   ryujin      main.cc ryujin_equation_dispatch.cc, linked against libryujin
#
# ryujin_add_hyperbolic_description(<name> TYPE <type> HEADER <header>)
# creates obj_<name>_typed from the four files in typed/ and
# sets RYUJIN_HYPERBOLIC_DESCRIPTION=<type> and
# RYUJIN_DESCRIPTION_HEADER=<header> on it.
