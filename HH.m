


for n = 1:12
for x=-5:0.05:5
    H = zeros(1,n);
    H(1)=1;
    H(2)=2*x;
    for i=3:n
        H(i)= 2*x*H(i-1)-2*(i-2)*H(i-2);
    end
    

jie = 1
for i=1:n
    jie = jie*i;
end
N = sqrt(1/(sqrt(2*3.14159)*(2^n)*jie))

fai = N*exp(-x*x/2)*H(n)

hold on

subplot(6,2,n)
title(['n=', num2str(n-1), ' figure'])
plot(x,fai*fai,'b*')

hold off
end
end

