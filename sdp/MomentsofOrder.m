function [number] = MomentsofOrder(species,order)
% This function computes the number of moments of a certain order for a
% given system.
n=species;
d=order;
number=factorial(d+n-1)/(factorial(d)*factorial(n-1));
end

