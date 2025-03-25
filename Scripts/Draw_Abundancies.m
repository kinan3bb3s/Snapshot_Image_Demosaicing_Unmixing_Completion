G_WNMF_cube=reshape(G_WNMF,[sz(1),sz(2),r]);
for i=1:r
    figure;imagesc(G_WNMF_cube(:,:,i));
end


G_PPID_cube=reshape(G_PPID,[sz(1),sz(2),r]);
for i=1:r
    figure;imagesc(G_PPID_cube(:,:,i));
end

for i=1:r
    figure;imagesc(G_HS_Cube(:,:,i));
end

