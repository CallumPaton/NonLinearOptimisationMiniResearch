function [alpha] = goldensearchMR(LowerBound, UpperBound, f, searchdirection,x,mu)
sympref('FloatingPointOutput',true);
format long
goldenratio = 0.618;
error = Inf;
errtol = 1e-5;
maxit = 100;
it = 0;

A = x+LowerBound*searchdirection;
B = x+UpperBound*searchdirection;

while abs(error)>errtol
Anew = A + (B-A)/goldenratio;
Bnew = B + (B-A)/goldenratio;

f1 = f(Anew,mu);
f2 = f(Bnew,mu);

if f1>f2
    B = Bnew;
elseif f1<f2
    A = Anew;
else
    A = Anew;
    B = Bnew;    
end
error = abs(B-A);
it = it +1;
if it>maxit
    fprintf('No Convergence to an optimal step size, alpha, after %d steps\n',it)
    break
end
end

argmin  = (A+B)/2;
alphad = argmin-x;
alpha = alphad./searchdirection;
alpha = alpha(1);

end