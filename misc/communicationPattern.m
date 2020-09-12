x = [-1, 0, 1];
y = [-1, 0, 1];
z = [-1, 0, 1];

[Y, X, Z] = meshgrid(y, x, z);

X = X(:);
Y = Y(:);
Z = Z(:);

scatter3(X, Y, Z);
hold on;

for x = [-1, 0, 1]; plot3([ x,  x], [-1,  0], [ 0, -1], 'r'); end
for x = [-1, 0, 1]; plot3([ x,  x], [-1,  1], [ 1, -1], 'r'); end
for x = [-1, 0, 1]; plot3([ x,  x], [ 0,  1], [ 1,  0], 'r'); end

for y = [-1, 0, 1]; plot3([-1,  0], [ y,  y], [ 0, -1], 'g'); end
for y = [-1, 0, 1]; plot3([-1,  1], [ y,  y], [ 1, -1], 'g'); end
for y = [-1, 0, 1]; plot3([ 0,  1], [ y,  y], [ 1,  0], 'g'); end

for z = [-1, 0, 1]; plot3([-1,  0], [ 0, -1], [ z,  z], 'b'); end
for z = [-1, 0, 1]; plot3([-1,  1], [ 1, -1], [ z,  z], 'b'); end
for z = [-1, 0, 1]; plot3([ 0,  1], [ 1,  0], [ z,  z], 'b'); end

hold off;