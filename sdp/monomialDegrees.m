function degrees = monomialDegrees(numVars, maxDegree)
if numVars==1
    degrees = (0:maxDegree).';
    return;
end
degrees = cell(maxDegree+1,1);
k = numVars;
for n = 0:maxDegree
    dividers = flipud(nchoosek(1:(n+k-1), k-1));
    degrees{n+1} = [dividers(:,1), diff(dividers,1,2), (n+k)-dividers(:,end)]-1;
end
degrees = cell2mat(degrees);