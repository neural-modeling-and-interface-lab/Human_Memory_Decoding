function P = SpikeTensor2BSplineFeatureMatrix(SpikeTensor,m,d)
% Convert spike tensor to B-spline feature matrix
% Get tensor dimension
[NP,L,N] = size(SpikeTensor);

% Generate B-splines
b = bspline(d+1,m+2,L);
Nb = m+d+1;

P = zeros(NP,Nb*N);

for i = 1:NP
    PSpike = squeeze(SpikeTensor(i,:,:))';
    PSplineM = PSpike*b;
    PSplineV = reshape(PSplineM',Nb*N,1)';
    P(i,:) = PSplineV;
end


