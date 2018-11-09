function [List] = UniqueElementsofOrder(zUniqueList,species,order)
% This function computes the number of moments of a certain order for a
% given system.
n=species;
d=order;
if d==0
    List=zUniqueList(1);
else
    orderBegin=TotalMomentsUptoOrder(n,d-1);
    orderEnd=TotalMomentsUptoOrder(n,d);
    List=zUniqueList(orderBegin+1:orderEnd);
end 
end

