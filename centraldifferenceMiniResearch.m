function [gradient] = centraldifferenceMiniResearch(f, x,mu)
h = 0.0001;

gradient = [];
for i = 1:length(x)
    input1 = x;
    input2 = x;
    input1(i) = x(i)+h;
    input2(i) = x(i)-h;
    gradient(end+1) = (f(input1,mu)-f(input2,mu))/(2*h);
end
%% GIHSGAIJCGIV

