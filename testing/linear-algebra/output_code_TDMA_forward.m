function output_code_TDMA_forward(M, rhs, x, filename)
    fid = fopen(filename,'w');
    [N, ~] = size(M);
    N = N/2;
    fprintf(fid, 'LINSYS_block_tridiag<__buffertype*, 2> linear_system;\n');
	fprintf(fid, 'int N = %d;\n', N);
	fprintf(fid, 'makesys<2>(&linear_system, N);\n');
    fprintf(fid, '//BEGIN MATLAB GENERATED CODE\n');
    for i = 1:N
        fprintf(fid, 'linear_system.rhs[0].base[%d] = %.15f;\n', i-1, rhs(i));
    end
    for i = 1:N
        fprintf(fid, 'linear_system.rhs[1].base[%d] = %.15f;\n', i-1, rhs(i+N));
    end
    for i = 1:N
        fprintf(fid, 'linear_system.block_matrices[0].diag.base[%d] = %.15f;\n', i-1, M(i,i));
    end
    for i = 1:N
        fprintf(fid, 'linear_system.block_matrices[1].diag.base[%d] = %.15f;\n', i-1, M(i,i+N));
    end
    for i = 1:N
        fprintf(fid, 'linear_system.block_matrices[2].diag.base[%d] = %.15f;\n', i-1, 0.0);
    end
    for i = 1:N
        fprintf(fid, 'linear_system.block_matrices[3].diag.base[%d] = %.15f;\n', i-1, M(i+N,i+N));
    end
    
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[0].sub.base[%d] = %.15f;\n', i-1, M(i+1,i));
    end
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[1].sub.base[%d] = %.15f;\n', i-1, M(i+1,i+N));
    end
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[2].sub.base[%d] = %.15f;\n', i-1, 0);
    end
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[3].sub.base[%d] = %.15f;\n', i-1, M(i+N+1,i+N));
    end
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[0].sup.base[%d] = %.15f;\n', i-1, M(i,i+1));
    end
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[1].sup.base[%d] = %.15f;\n', i-1, M(i,i+1+N));
    end
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[2].sup.base[%d] = %.15f;\n', i-1, 0);
    end
    for i = 1:N-1
        fprintf(fid, 'linear_system.block_matrices[3].sup.base[%d] = %.15f;\n', i-1, M(i+N,i+N+1));
    end
    
    fprintf(fid, 'HYBRID::solve_one_way_coupled<__hybrid, 2>(&linear_system);\ndouble error = 0;\n');
    for i = 1:N
        fprintf(fid, 'error += (linear_system.rhs[0].base[%d] - (%.15f));\n', i-1, x(i));
    end
    for i = 1:N
        fprintf(fid, 'error += (linear_system.rhs[1].base[%d] - (%.15f));\n', i-1, x(i+N));
    end
    
    fprintf(fid, 'killsys(&linear_system);\n');
    fprintf(fid, '//END MATLAB GENERATED CODE\n');
    fclose(fid);

end