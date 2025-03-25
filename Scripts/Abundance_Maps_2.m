G=reshape(G_HS,[sz(1),sz(2),r]);
figure;subplot(1,3,1);imagesc(G(:,:,1));
subplot(1,3,2);imagesc(G(:,:,2));
subplot(1,3,3);imagesc(G(:,:,3));

G_VCA=reshape(G_WNMF,[sz(1),sz(2),r]);
figure;subplot(1,3,1);imagesc(G_VCA(:,:,1));
subplot(1,3,2);imagesc(G_VCA(:,:,3));
subplot(1,3,3);imagesc(G_VCA(:,:,2));


G_VCA1=reshape(G_WNMF1,[sz(1),sz(2),r]);
figure;subplot(1,3,1);imagesc(G_VCA1(:,:,3));
subplot(1,3,2);imagesc(G_VCA1(:,:,1));
subplot(1,3,3);imagesc(G_VCA1(:,:,2));

G_VCA2=reshape(G_WNMF2,[sz(1),sz(2),r]);
figure;subplot(1,3,1);imagesc(G_VCA2(:,:,2));
subplot(1,3,2);imagesc(G_VCA2(:,:,3));
subplot(1,3,3);imagesc(G_VCA2(:,:,1));

G_PPID_H=reshape(G_PPID,[sz(1),sz(2),r]);
figure;
subplot(1,3,1);imagesc(G_PPID_H(:,:,2));
subplot(1,3,2);imagesc(G_PPID_H(:,:,3));
subplot(1,3,3);imagesc(G_PPID_H(:,:,1));