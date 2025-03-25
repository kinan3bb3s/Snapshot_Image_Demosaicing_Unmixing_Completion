v1=zeros(1,10000);
v1(1:2000)=1;
for i=2001:4000 
 v1(i)=round(rand(1,1),1)+0.000001;
end

v2=zeros(1,10000);
for i=2001:4000 
 v2(i)=round(rand(1,1),1)+0.000001;
end
v2(4001:6000)=1;
for i=6001:7000
 v2(i)=round(rand(1,1),1)+0.000001;
end

v3=zeros(1,10000);
for i=6001:7000
 v3(i)=round(rand(1,1),1)+0.000001;
end

v3(7001:10000)=1;
v1=v1';v2=v2';v3=v3';
G_HS=[v1,v2,v3];

for i=1:10000
    tmp=G_HS(i,:);
    idx=tmp==0;
    out=sum(idx(:));
    if out<2
        G_HS(i,:)=ScaleRows(tmp);
    end
end

for i=1:10000
    tmp=G_HS(i,:);
    out=sum(tmp);
    if out>1
        for j=1:3
            if(tmp(j)~=1)
                tmp(j)=0;
            end
        end
        G_HS(i,:)=tmp;
    end
end