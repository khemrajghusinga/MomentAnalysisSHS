# Moment Analysis of Stochastic Hybrid Systems Using Semidefinite Programming 
## Authors: Khem Raj Ghusinga, Andrew Lamperski, and Abhyudai Singh

This repository contains codes related to the following paper:

Ghusinga, Khem Raj, Andrew Lamperski, and Abhyudai Singh. "Moment analysis of stochastic hybrid systems using semidefinite programming." Automatica 112: 108634, 2020.

The code to simulate the Stochastic Hybrid System (SHS) in Example 1 of the paper is provided in the folder "simulateSHSrecast". The code to compute lower and upper bounds on the moments using semidefinite program in the Example 1 of the paper is in the folder "sdp".

Requirements:
- To simulate the SHS, python 2.7 is required
- To use the SDP code, the YALMIP wrapper (https://yalmip.github.io) along with SDPA-GMP solver (http://sdpa.sourceforge.net) is required.
