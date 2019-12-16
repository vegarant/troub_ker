function out_str = generate_poly_heav(ph)
    
    degree = ph.degree;
    discon = ph.discon;
    coeff = ph.coeff;
    bias = ph.bias;
    a = ph.a;
    b = ph.b;
    
    str_cell = cell([degree-1, 1]);
    out_str = '';

    if degree > 0
        str_cell{1} = sprintf('%18.16f.*(x - %18.16f)', coeff(degree+1), coeff(1));
    end

    for i = 2:degree
        str_cell{i} = sprintf('(x - %18.16f)', coeff(i));
    end

    for i = 1:degree-1
        out_str = [out_str, str_cell{i}, '.*'];
    end

    out_str = [out_str, str_cell{degree}, sprintf(' + %18.16f', bias)]; 
    
    for i = 1:discon
        str_cell{i} = sprintf('%18.16f.*(x > %18.16f)', b(i), a(i));
    end
    for i = 1:discon
        out_str = [out_str, ' + ', str_cell{i}];
    end

end
