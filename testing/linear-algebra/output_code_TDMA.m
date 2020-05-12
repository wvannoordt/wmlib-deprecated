function output_code_TDMA(M, rhs, x, filename)
    fid = fopen(filename,'w');
    [N, ~] = size(M);
    fprintf(fid, 'MAT_tridiag<double*> lhs;\n');
    fprintf(fid, 'Buffer<double*> rhs;\n');
	fprintf(fid, 'int N = %d;\n', N);
    fprintf(fid, '//BEGIN MATLAB GENERATED CODE\n');
    fprintf(fid, 'makebuf(&rhs, N);\n');
	fprintf(fid, 'makebuf(&(lhs.diag), N);\n');
	fprintf(fid, 'makebuf(&(lhs.sup),  N-1);\n');
	fprintf(fid, 'makebuf(&(lhs.sub),  N-1);\n');
    fprintf(fid, 'lhs.dim = N;\n');
    for i = 1:N
        fprintf(fid, 'rhs.base[%d] = %.15f;\n', i-1, rhs(i));
    end
    for i = 1:N
        fprintf(fid, 'lhs.diag.base[%d] = %.15f;\n', i-1, M(i,i));
    end
    for i = 1:N-1
        fprintf(fid, 'lhs.sub.base[%d] = %.15f;\n', i-1, M(i+1,i));
    end
    for i = 1:N-1
        fprintf(fid, 'lhs.sup.base[%d] = %.15f;\n', i-1, M(i,i+1));
    end
    
    fprintf(fid, 'HYBRID::solve_thomas<__hybrid>(&lhs, &rhs);\n');
    fprintf(fid, 'double error = 0;\n');
    for i = 1:N
        fprintf(fid, 'error += (rhs.base[%d] - (%.15f));\n', i-1, x(i));
    end

    fprintf(fid, 'killbuf(&rhs);\n');
	fprintf(fid, 'killbuf(&(lhs.diag));\n');
	fprintf(fid, 'killbuf(&(lhs.sup));\n');
	fprintf(fid, 'killbuf(&(lhs.sub));\n');
    fprintf(fid, '//END MATLAB GENERATED CODE\n');
    fclose(fid);

end