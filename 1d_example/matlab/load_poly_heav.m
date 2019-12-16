function ph = load_poly_heav(fname)
    
    poly_heav = load(fname);
    function_str = generate_poly_heav_string(poly_heav);
    poly_heav.function_handle = str2func(['@(x) ', function_str]);
    poly_heav.function_str = function_str;
    ph = poly_heav;


end
