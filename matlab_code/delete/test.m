

nJoints = 10;
observedJoints = [ 2  8  3 5]

for j=1:nJoints
    if sum(j==observedJoints)~=0
        fprintf('index is %g\n', j)
    else
        disp('fillzero')
    end
end


% 
% j == observedJoints
% 
% 
% 
% observedJoints == 