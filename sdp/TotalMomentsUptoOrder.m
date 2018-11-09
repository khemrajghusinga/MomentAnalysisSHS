function [number] = TotalMomentsUptoOrder(species,order)
% This function computes the number of moments of a certain order for a
% given system.
n=species;
d=order;
number=factorial(d+n)/(factorial(d)*factorial(n));
end

