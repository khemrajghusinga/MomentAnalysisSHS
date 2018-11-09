# Moment Analysis of Stochastic Hybrid Systems Using Semidefinite Programming 
## Authors: Khem Raj Ghusinga, Andrew Lamperski, and Abhyudai Singh
In this repository, we provide the code to simulate the SHS in Example 1 of the paper (see folder "simulateSHSrecast"). We also provide the semidefinite program to compute lower and upper bounds on the moments in the Example 1 of the paper (see folder "sdp").

Requirements:
- To simulate the SHS, python 2.7 is required
- To use the SDP code, the YALMIP wrapper (https://yalmip.github.io) along with SDPA-GMP solver (http://sdpa.sourceforge.net) is required.
