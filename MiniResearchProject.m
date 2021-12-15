%% Mini Research Project
format short

%% Initial Values 

LowerBound = -5;
UpperBound = 5;
err = Inf;
it = 1;
t = 1;
mu = 0.5;

%% Objective Function
x_initial = [-4,6];
f= @(x) (x(1)-1).^2+x(2);

maxit = 200;
epsilon = 1e-8;

%% Quadratic Penalty Function and Constraints
g = @(x,mu) (1/(2*mu))*((x(1).^2)/10+(x(2).^2)/1.2-2)^2;
%% Combined Non-Linear Optimisation Problem
phi = @(x,mu) f(x)+g(x,mu);
%% Initialising log and solution files
fopen('lognesterov.txt','w');
fopen('solutionnesterov.txt','w');
diary 'lognesterov.txt'

%% Calculating initial gradient and search direction.
x_guesses = x_initial';
[gradient] = centraldifferenceMiniResearch(phi, x_initial,mu);
gradient = gradient';
searchdirection = -gradient;

x = x_initial'; 

%% Plotting Steps
[X,Y] = meshgrid( linspace(-10,10,50), linspace(-10,10,50));
%h = @(x,y) (1-x).^2 + 100*(y-x.^2).^2
h = @(x,y) (x-1).^2+y;
contourf(X,Y,h(X,Y),20,'linestyle','none');
hold on
contour(X,Y,(X.^2)/10+(Y.^2)/1.2,[2,2],'w','LineWidth',2)

%% Printing initial set up
fprintf('Initial function to solve,\n')
fprintf('f(x) =')
disp(f)
fprintf('With initial starting vector, \n')
fprintf('x1 = [')
fprintf('%g, ', x_initial(1:end-1));
fprintf('%g]\n', x_initial(end));
fprintf('Intial direction vector, d_1 = -g_1\n')
fprintf('d1 = [')
fprintf('%g, ', searchdirection(1:end-1));
fprintf('%g]\n', searchdirection(end));


%% Nesterov Loop
run = true;
while run
      
      % Find optimal alpha using golden search
      [alpha] = goldensearchMR(LowerBound, UpperBound, phi, searchdirection,x,mu);
      fprintf('alpha_(%d), found from golden search = %f\n',it, alpha)
      
      % Calculate xj+1 = xj + alpha d
      x_next = x + alpha*searchdirection;
      
      % Nesterov Accelerated Gradient Step
      t_next = (1+sqrt(1+4*t^2))/2;
      y_next = x_next + ((t-1)/t_next)*(x_next-x);
      x_guesses = [x_guesses y_next];
      it = it+1;
      if it>maxit
        fprintf('No Optimal Solution found after %d iterations\n',it)
        break
      end
      fprintf('Now we can calculate next solution, \n')
      fprintf('x_(%d) = x_(%d) + alpha_(%d) * d_(%d)  = [',it,it-1,it-1,it-1)
      fprintf('%g, ', y_next(1:end-1));
      fprintf('%g]\n', y_next(end));
      
      % Check Stopping Criterion
      f_next  = phi(y_next,mu);
      f_old = phi(x,mu);
      
      
      err = abs(f_next - f_old);
      
      if err < epsilon
          diary off
          type 'log.txt'
          diary 'solutionnesterov.txt'
          fprintf('|f(x_%d)-f(x_%d)|< epsilon \n',it, it-1)
          fprintf('Converged to solution\n\n')
          fprintf('Optimal Value of function = %d\n\n', f_next)
          fprintf('With solution vector\n\n')
          fprintf('x_(%d) = [',it)
          fprintf('%g, ', y_next(1:end-1));
          fprintf('%g]\n', y_next(end));
          diary off
          type 'solutionnesterov.txt'
          run = false;
          break
      else
          fprintf('Solution not optimal\n')
      end
      
      % Calculate New Gradient
      [gradient_next] = centraldifferenceMiniResearch(phi, y_next,mu);
      gradient_next = gradient_next';
      
      % Calculate beta using Fletcher-Reeves form
      beta = (gradient_next'*gradient_next)/(gradient'*gradient);
      
      % Calculate new search direction
      searchdirection_next = -gradient_next + beta*searchdirection;
      fprintf('New direction vector, d_(%d) = -g_(%d) + beta*d_(%d)\n',it,it,it-1)
      
      % Reset Values
      x = y_next;
      searchdirection = searchdirection_next;
      gradient = gradient_next;
      %t = t_next;
end

plot(x_guesses(1,:),x_guesses(2,:),'-ow')
plot(x_guesses(1,end),x_guesses(2,end),'kx','LineWidth', 2)
xlabel('$x_1$','interpreter','latex','FontSize',14)
ylabel('$x_2$','interpreter','latex','FontSize',14)
ylim([-5 5])
xlim([-5 5])
title('$\mu = 5$','interpreter','latex','FontSize',14)




