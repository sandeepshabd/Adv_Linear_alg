% SVD-based image compression for grayscale image

% Step 1: Read color image and convert to grayscale
img_color = imread('peppers.png');  % Replace with your image
img_gray = rgb2gray(img_color);     % Convert to grayscale
A = im2double(img_gray);            % Convert to double

% Step 2: Compute SVD
[U, S, V] = svd(A);
singular_vals = diag(S);
total_energy = sum(singular_vals.^2);

% Step 3: Find optimal k (retain 99% of energy)
energy = 0;
k_opt = 0;
for k = 1:length(singular_vals)
    energy = energy + singular_vals(k)^2;
    if energy / total_energy >= 0.99
        k_opt = k;
        break;
    end
end

% Step 4: Reconstruct image using optimal k
A_k = U(:,1:k_opt) * S(1:k_opt,1:k_opt) * V(:,1:k_opt)';

% Step 5: Display original and compressed images
figure;
subplot(1,2,1);
imshow(img_gray);
title('Original Grayscale Image');

subplot(1,2,2);
imshow(A_k);
title(['Compressed Image (k = ' num2str(k_opt) ')']);

% Step 6: Plot singular values and reconstruction error
errors = zeros(1, length(singular_vals));
for k = 1:length(singular_vals)
    A_tmp = U(:,1:k) * S(1:k,1:k) * V(:,1:k)';
    errors(k) = norm(A - A_tmp, 'fro');
end

figure;
subplot(1,2,1);
plot(singular_vals, 'b-o');
title('Singular Values');
xlabel('Index'); ylabel('Value');

subplot(1,2,2);
plot(errors, 'r');
title('Reconstruction Error vs. Rank');
xlabel('Rank (k)'); ylabel('Frobenius Norm Error');

% Step 7: Show compression ratio
[m, n] = size(A);
original_vals = m * n;
compressed_vals = k_opt * (m + n + 1);  % U + V + S
ratio = original_vals / compressed_vals;

disp(['✅ Optimal k = ' num2str(k_opt)]);
disp(['📦 Compression Ratio = ' num2str(ratio, '%.2f')]);
