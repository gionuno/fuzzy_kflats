function [W,b] = fuzzy_kflats_clustering(X,K,D,g,T)
    N = size(X,1);
    M = size(X,2);

    W = randn(K,M,D);
    b = randn(K,D);
    for k = 1:K
        for d = 1:D
            W(k,:,d) = W(k,:,d)/norm(W(k,:,d));
        end
    end
    C = zeros(N,K);
    for t = 1:T
        disp(t);
        for k = 1:K
            R = X*reshape(W(k,:,:),[M,D])-repmat(b(k,:),[N,1]);
            C(:,k) = arrayfun(@(b) (norm(R(b,:)).^2+1e-8).^(-1.0/(g-1.0)),1:N);
        end
        sC = sum(C,2);
        p = C ./ repmat(sC,[1,K]);
        for k = 1:K
            pg = p(:,k).^g;
            pg = pg / sum(pg);
            
            Y = X-repmat(pg'*X,[N,1]);
            [V,~] = eig(Y'*diag(pg)*Y+1e-8*eye(M));
            W(k,:,:) = real(V(:,1:D));
            b(k,:) = p(:,k)'*X*reshape(W(k,:,:),[size(X,2),D]);
        end
    end
end