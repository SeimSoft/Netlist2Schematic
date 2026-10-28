"""Textbook circuits with hand-made placements following the usual schematic rules:

- signal flow left to right (input left, output right),
- supply at the top, ground at the bottom (ground symbols point down),
- parts on a common line (same row / column) wherever they connect, no wire crossings,
- vertical shunt parts (to ground) hang below the signal line, series parts are horizontal.

Placement: part name -> (x, y, orientation). Orientation 0 = as drawn by the symbol
(two-terminal parts horizontal, pin 0 left), 1 = quarter turn (pin 0 on top),
2 = half turn, 3 = three quarters (pin 0 at the bottom), 4-7 = mirrored then turned
(6 flips a transistor vertically). "port:<net>" are the label parts of nets with
several pins; a net with a single pin gets its label right at the pin.
"""

CIRCUITS: dict[str, tuple[str, dict[str, tuple[int, int, int]]]] = {}

# ------------------------------------------------------------------ batch 1: small

CIRCUITS["rc_lowpass"] = (
    """
.subckt rc_lowpass in out
R1 in out 1k
C1 out 0 100n
.ends
""",
    {"R1": (0, 0, 0), "C1": (4, 3, 1), "port:out": (9, 0, 2)},
)

CIRCUITS["divider"] = (
    """
.subckt divider out
V1 a 0 5
R1 a out 10k
R2 out 0 10k
.ends
""",
    {"V1": (0, 6, 1), "R1": (5, 0, 1), "R2": (5, 6, 1), "port:out": (10, 3, 2)},
)

CIRCUITS["led_switch"] = (
    """
.subckt led_switch in vcc
R1 vcc a 330
D1 a c LED
Q1 c b 0 NPN
Rb in b 10k
.model LED D
.model NPN NPN
.ends
""",
    {"R1": (4, -8, 1), "D1": (4, -2, 1), "Q1": (3, 4, 0), "Rb": (-3, 4, 0)},
)

CIRCUITS["pi_filter"] = (
    """
.subckt pi_filter in out
C1 in 0 1u
L1 in out 10u
C2 out 0 1u
.ends
""",
    {"port:in": (-5, 0, 0), "C1": (0, 3, 1), "L1": (4, 0, 0), "C2": (8, 3, 1), "port:out": (13, 0, 2)},
)

CIRCUITS["common_emitter"] = (
    """
.subckt common_emitter in out vcc
Cin in b 1u
R1 vcc b 47k
R2 b 0 10k
Rc vcc c 4.7k
Q1 c b e NPN
Re e 0 1k
Ce e 0 10u
Cout c out 1u
.model NPN NPN
.ends
""",
    {
        "Cin": (2, 6, 0),
        "R1": (6, 0, 1),
        "R2": (6, 12, 1),
        "Q1": (10, 6, 0),
        "Rc": (11, 0, 1),
        "Re": (11, 12, 1),
        "Ce": (15, 12, 1),
        "Cout": (16, 3, 0),
    },
)

CIRCUITS["mosfet_switch"] = (
    """
.subckt mosfet_switch in vcc
Rload vcc d 10
D1 d vcc DFLY
M1 d g 0 NMOS
Rg in g 100
Rpd g 0 100k
.model DFLY D
.model NMOS NMOS
.ends
""",
    {"Rload": (9, 0, 1), "D1": (13, 0, 3), "M1": (8, 6, 0), "Rg": (2, 6, 0), "Rpd": (5, 10, 1)},
)

# ------------------------------------------------------------------ batch 2: larger
# Port names follow the project convention: signals carry a direction suffix (_i, _o,
# _ai, _ao), supply rails are upper case (VCC, VDD).

CIRCUITS["h_bridge"] = (
    """
.subckt h_bridge in1_i in2_i in3_i in4_i VCC
R1 in1_i b1 1k
Q1 ma b1 VCC PNP
R2 in2_i b2 1k
Q2 ma b2 0 NPN
R3 in3_i b3 1k
Q3 mb b3 VCC PNP
R4 in4_i b4 1k
Q4 mb b4 0 NPN
D1 ma VCC DF
D2 0 ma DF
D3 mb VCC DF
D4 0 mb DF
Lm ma mb 1m
.model PNP PNP
.model NPN NPN
.model DF D
.ends
""",
    {
        "R1": (2, 0, 0), "Q1": (8, 0, 6), "R2": (2, 10, 0), "Q2": (8, 10, 0),
        "D1": (13, 0, 3), "D2": (13, 10, 3), "Lm": (18, 3, 0),
        "D3": (23, 0, 3), "D4": (23, 10, 3),
        "Q3": (28, 0, 2), "R3": (34, 0, 2), "Q4": (28, 10, 4), "R4": (34, 10, 2),
    },
)

CIRCUITS["ldo"] = (
    """
.subckt ldo vref_ai out_ao VDD
X1 vref_ai fb gate VDD 0 EA
MP out_ao gate VDD PMOS
Rf1 out_ao fb 100k
Rf2 fb 0 100k
Cout out_ao 0 1u
Rload out_ao 0 1k
.model PMOS PMOS
.ends
.subckt EA INN_AI INP_AI OUT_AO VCC VEE
E1 OUT_AO 0 INP_AI INN_AI 1e5
.ends
""",
    {"X1": (0, 0, 0), "MP": (8, 0, 6), "Rf1": (9, 7, 1), "Rf2": (9, 13, 1), "Cout": (14, 7, 1), "Rload": (18, 7, 1), "port:out_ao": (23, 4, 2)},
)
