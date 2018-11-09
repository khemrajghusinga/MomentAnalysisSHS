function [Meven,MevenLength,Modd,ModdBounded,ModdLength]=MomentMatrixIndex(numVars,maxDegree)
%numVars=3; maxDegree=3;
mu=monomialDegrees(numVars, maxDegree);

if mod(maxDegree,2) == 0
    deven=maxDegree/2;
    dodd=deven-1;
else 
    deven=(maxDegree-1)/2;
    %deven=(maxDegree+1)/2;
    dodd=deven;
end

MevenLength=TotalMomentsUptoOrder(numVars,deven);
ModdLength=TotalMomentsUptoOrder(numVars,dodd);

Meven=zeros(MevenLength);
Meven(:,1)=1:MevenLength;
Meven(1,:)=1:MevenLength;
for i=1:MevenLength
   for j=1:MevenLength
       [~,a]=ismember(mu(i,:)+mu(j,:),mu,'rows');
       Meven(i,j)=a;
   end
end

numOddMats=numVars;

Modd=zeros(ModdLength,ModdLength,numOddMats);
ModdBounded=zeros(ModdLength,ModdLength,numOddMats);
for idx=1:numOddMats
    Modd(:,1,idx)=1:ModdLength;
    ModdBounded(:,1,idx)=1:ModdLength;
    Modd(1,:,idx)=1:ModdLength;
    ModdBounded(1,:,idx)=1:ModdLength;
end 
oddsums=eye(numOddMats);
for oddidx=1:numOddMats
    for i=1:ModdLength
        for j=1:ModdLength
            [~,a]=ismember(mu(i,:)+mu(j,:)+oddsums(oddidx,:),mu,'rows');
            Modd(i,j,oddidx)=a;
            [~,b]=ismember(mu(i,:)+mu(j,:),mu,'rows');
            ModdBounded(i,j,oddidx)=b;
        end
    end
end
