m = randi(10,100,8000);
m1 = m>0;

tbl = zeros(size(m,2));

for i=1:size(m,2)
    for j=1:size(m,2)
        tbl(i,j) = m(:,i)'*m(:,j)/size(m,1);
    end
end