%% check some basic functions
function draft

%% check kernel function

dx = 0.01;
kh = 6;
kshow = 2.5;

h = kh*dx;
r = (-kshow*h:0.001:kshow*h)';
W = zeros(size(r));
grad_W = zeros(size(r));
lapl_W = zeros(size(r));
% W = cal_kn_cubicspline(r, h);
% grad_W = cal_kn_grad_cubicspline(r, h);
% lapl_W = cal_kn_lapl_cubicspline(r, h);
W = cal_kn_WendlandC2(r, h);
grad_W = cal_kn_grad_WendlandC2(r, h);
lapl_W = cal_kn_lapl_WendlandC2(r, h);

figure(1)
cla; hold on; grid on; axis tight;
xlabel("q"); ylabel("f(q)");
plot(r/h, W, 'b')
plot(r/h, grad_W, 'g')
% plot(r/h, lapl_W, 'r')
legend('W', '∇W', '∇2W');

% end of main function
flag_end = 1;

function W = cal_kn_cubicspline(r, h)
qq = abs(r/h);
kd = [4/3/h; 40/7/pi/h^2; 8/pi/h^3];
v = zeros(size(qq));
for i = 1:size(r, 1)
    q = qq(i);
    if q >= 0 && q <= 0.5
        v(i) = 6*(q^3-q^2)+1;
    elseif q > 0.5 && q <= 1
        v(i) = 2*(1-q)^3;
    else
        v(i) = 0;
    end
end
v = kd(2)*v;
W = v;

function grad_W = cal_kn_grad_cubicspline(r, h)
qq = abs(r/h);
kd = [4/3/h; 40/7/pi/h^2; 8/pi/h^3];
v = zeros(size(qq));
for i = 1:size(r, 1)
    q = qq(i);
    if q >= 0 && q <= 0.5
        v(i) = 6*(3*q^2-2*q);
    elseif q > 0.5 && q <= 1
        v(i) = -6*(1-q)^2;
    else
        v(i) = 0;
    end
end
v = kd(2)*v;
% grad_W = v.*sign(r)/h;
grad_W = v.*sign(r);

function lapl_W = cal_kn_lapl_cubicspline(r, h)
qq = abs(r/h);
kd = [4/3/h; 40/7/pi/h^2; 8/pi/h^3];
v = zeros(size(qq));
for i = 1:size(r, 1)
    q = qq(i);
    if q >= 0 && q <= 0.5
        v(i) = 6*(6*q-2);
    elseif q > 0.5 && q <= 1
        v(i) = 12*(1-q);
    else
        v(i) = 0;
    end
end
v = kd(2)*v;
% lapl_W = v/h^2;
lapl_W = v;

function W = cal_kn_WendlandC2(r, h)
qq = abs(r/h);
kd = [5/(8*h); 7/(4*pi*h^2); 21/(2*pi*h^3)];
v = zeros(size(qq));
for i = 1:size(r, 1)
    q = qq(i);
    if q >= 0 && q <= 2
        v(i) = (1-0.5*q)^4*(1+2*q);
    else
        v(i) = 0;
    end
end
v = kd(2)*v;
W = v;

function grad_W = cal_kn_grad_WendlandC2(r, h)
qq = abs(r/h);
kd = [5/(8*h); 7/(4*pi*h^2); 21/(2*pi*h^3)];
v = zeros(size(qq));
for i = 1:size(r, 1)
    q = qq(i);
    if q >= 0 && q <= 2
        v(i) = -5*q*(1-0.5*q)^3;
    else
        v(i) = 0;
    end
end
v = kd(2)*v;
% grad_W = v.*sign(r)/h;
grad_W = v.*sign(r);

function lapl_W = cal_kn_lapl_WendlandC2(r, h)
qq = abs(r/h);
kd = [5/(8*h); 7/(4*pi*h^2); 21/(2*pi*h^3)];
v = zeros(size(qq));
for i = 1:size(r, 1)
    q = qq(i);
    if q >= 0 && q <= 2
        v(i) = -5*(1-2*q)*(1-0.5*q)^2;
    else
        v(i) = 0;
    end
end
v = kd(2)*v;
% lapl_W = v/(h*h);
lapl_W = v;


