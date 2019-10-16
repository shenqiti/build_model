function y; %函数声明
figure; %建立新的图形文件
p=-pi:pi/100:pi; t= 0:2*pi/150:pi;%设置p,t变化范围和步长
[P,T]=meshgrid(p,t);theta=pi/2-P;phi=T;
R=Ys(theta,phi);%用定义的函数计算R值
[X,Y,Z]=sph2cart(T,P,R);%建立一个矩阵，并以变换后的T,P,R值为其赋值
mesh(X,Y,Z);%对矩阵[X，Y Z]做网格图
axis equal;xlabel('X'),ylabel('Y'),zlabel('Z');title('Ys')
function y=Ys(theta,phi);
y=1/8*sqrt(42/pi)*abs(sin(theta).*(5.*cos(theta).*cos(theta)-1).*sin(phi))%函数定义



%Y00=sqrt(1/4/pi)
%Y10=sqrt(3/4/pi)*abs(cos(theta))
%Y11=sqrt(3/4/pi)*abs(sin(theta).*cos(phi))
%Y1-1=sqrt(3/4/pi)*abs(sin(theta).*sin(phi))
%Y20=sqrt(5/16/pi)*abs(3.*cos(theta).*cos(theta)-1)
%Y21=sqrt(15/4/pi)*abs(sin(theta).*cos(theta).*cos(phi))
%Y2-1= sqrt(15/4/pi)*abs(sin(theta).*cos(theta).*sin(phi))
%Y22=sqrt(15/16/pi)*abs(sin(theta).*sin(theta).*sin(2*phi))
%Y2-2= sqrt(15/16/pi)*abs(sin(theta).*sin(theta).*cos(2*phi))
%Y30=1/4*sqrt(7/pi)*abs(5.*cos(theta).*cos(theta).*cos(theta)-3.*cos(theta))
%Y31=1/8*sqrt(42/pi)*abs(sin(theta).*(5.*cos(theta).*cos(theta)-1).*cos(phi))
%Y32=1/4*sqrt(105/pi)*abs(sin(theta).*sin(theta).*cos(theta).*cos(2*phi))
%Y3-1=1/8*sqrt(42/pi)*abs(sin(theta).*(5.*cos(theta).*cos(theta)-1).*sin(phi))
%Y3-2=1/4*sqrt(105/pi)*abs(sin(theta).*sin(theta).*cos(theta).*sin(2*phi))
%Y33=1/8*sqrt(70/pi)*abs(sin(theta).*sin(theta).*sin(theta).*cos(3*phi))
%Y3-3=1/8*sqrt(70/pi)*abs(sin(theta).*sin(theta).*sin(theta).*sin(3*phi))
