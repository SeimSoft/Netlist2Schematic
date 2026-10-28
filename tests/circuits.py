"""Small circuits shared by the tests."""

RC = """
.subckt rc vin vout gnd
R1 vin vout 1k
C1 vout gnd 1u
.ends
"""

# Valid hand placement of RC (deliberately not the policy's own proposal).
HUMAN = {"R1": (8, 8, 7), "C1": (13, 8, 6), "port:vout": (10, 6, 0)}

TWO_ICS = """
.subckt top vin vout gnd
X1 vin mid gnd CHIP
X2 mid vout gnd CHIP
R1 vin gnd 10k
.ends
.subckt CHIP IN OUT GND
R9 IN OUT 1
.ends
"""
