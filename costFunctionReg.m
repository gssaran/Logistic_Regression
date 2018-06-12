function [J, grad] = costFunctionReg(theta, X, y, lambda)
m = length(y); % number of training examples
J = 0;
h=sigmoid(X*theta);
n=length(theta);
J=1/m*(-y'*log(h)-(1-y)'*log(1-h))+lambda/(2*m)*(theta'*theta-theta(1)*theta(1));
grad = zeros(size(theta));
grad =1/m*(X'*(h-y));
grad(2:n,1)=grad(2:n,1)+lambda/m*theta(2:n,1);
end
