use pyo3::prelude::*;

#[pyclass]
struct lr_obj {
    #[pyo3(get, set)]
    w: f64,
    #[pyo3(get, set)]
    b: f64,
    #[pyo3(get, set)]
    j_hist: Vec<f64>,
    #[pyo3(get, set)]
    p_hist: Vec<Vec<f64>>,
}

#[pyclass]
struct Lrobj_multi {
    #[pyo3(get, set)]
    w: Vec<f64>,
    #[pyo3(get, set)]
    b: f64,
    #[pyo3(get, set)]
    j_hist: Vec<f64>
}

#[pymethods]
impl lr_obj {
    #[new]
    #[args(w = "0.0", b = "0.0", j_hist = "vec![]", p_hist = "vec![]")]
    fn new(w: f64, b: f64, j_hist: Vec<f64>, p_hist: Vec<Vec<f64>>) -> Self {
        lr_obj { w, b, j_hist, p_hist }
    }

    #[args(x = "0.0")]
    fn predict_x(&self, x: f64) -> f64 {
        self.w * x + self.b
    }
}

#[pymethods]
impl Lrobj_multi {
    #[new]
    #[args(w = "vec![]", b = "0.0", j_hist = "vec![]")]
    fn new(w: Vec<f64>, b: f64, j_hist: Vec<f64>) -> Self {
        Lrobj_multi { w, b, j_hist }
    }
}

/// Linear Regression
#[pyfunction]
fn compute_cost(x: Vec<f64>, y: Vec<f64>, w: f64, b: f64) -> f64 {
    let mut cost: f64 = 0.0;
    for i in 0..x.len() {
        let f_wb = w * x[i] + b;
        cost += (f_wb - y[i]).powi(2);
    }
    1.0 / (2.0 * x.len() as f64) * cost
}

#[pyfunction]
fn compute_gradient(x: Vec<f64>, y: Vec<f64>, w: f64, b: f64) -> (f64, f64) {
    let l = x.len();
    let mut dj_dw : f64 = 0.0;
    let mut dj_db : f64 = 0.0;

    for i in 0..l {
        let f_wb : f64 = w * x[i] + b;
        let dj_dw_i : f64 = (f_wb - y[i]) * x[i];
        let dj_db_i : f64 = f_wb - y[i];
        dj_dw += dj_dw_i;
        dj_db += dj_db_i;
    }

    (dj_dw / l as f64, dj_db / l as f64)
}

#[pyfunction]
fn gradient_descent(x: Vec<f64>, y: Vec<f64>, w_in: f64, b_in: f64, alpha: f64, num_iters: u64) -> PyResult<lr_obj> {
    let mut j_hist : Vec<f64> = vec![];
    let mut p_hist : Vec<Vec<f64>> = vec![];
    let mut b = b_in;
    let mut w = w_in;

    for i in 0..num_iters {
        let (dj_dw, dj_db) = compute_gradient(x.clone(), y.clone(), w, b);
        b -= alpha * dj_db;
        w -= alpha * dj_dw;
        
        if i < 100000 {
            j_hist.push(compute_cost(x.clone(), y.clone(), w, b));
            p_hist.push(vec![w, b]);
        }

        if i % (num_iters / 10) == 0 {
            let j_hist_last = j_hist[j_hist.len() - 1];
            println!("Iteration {i}: Cost {j_hist_last}  dj_dw: {dj_dw}  dj_db: {dj_db}  w: {w}  b: {b}");
        }
    }
    Ok(lr_obj::new(w, b, j_hist, p_hist))
}

// Writing for Multi-variate Linear Regression
#[pyfunction]
fn predict_single_value(x: Vec<f64>, w: Vec<f64>, b: f64) -> f64 {
    w.iter().zip(x.iter()).fold(b, |acc, (&wi, &xi)| acc + wi * xi)
}

#[pyfunction]
fn compute_cost_multi(x: Vec<Vec<f64>>, y: Vec<f64>, w: Vec<f64>, b: f64) -> f64 {
//x.iter().zip(y.iter()).fold(0.0, |acc, (&ai, &bi)| acc + (predict_single_value(ai, w, b) - bi).powi(2)) / (2.0*x.len() as f64)
    let mut cost : f64 = 0.0;
    for (ai, bi) in x.iter().zip(y.iter()) {
        let err = predict_single_value(ai.to_vec(), w.clone(), b) - bi;
        let err_sq = err.powi(2);
        cost += err_sq;
    }
    cost / (2.0 * x.len() as f64)
}

#[pyfunction]
fn compute_gradient_multi(x: Vec<Vec<f64>>, y: Vec<f64>, w: Vec<f64>, b: f64) -> (Vec<f64>, f64) {
    let mut dj_dw : Vec<f64> = vec![0.0; x[0].len()];
    let mut dj_db : f64 = 0.0;
    let m = x.len() as f64;
    for (ai, bi) in x.iter().zip(y.iter()) {
        let err = predict_single_value(ai.to_vec(), w.clone(), b) - bi;
        for i in 0..x[0].len() {
            dj_dw[i]+=err*ai[i];
            dj_db += err;
        }
    }
    dj_dw = dj_dw.iter().map(|&x| x / m).collect();
    dj_db/=m;
    (dj_dw, dj_db)
}

#[pyfunction]
fn gradient_descent_multi(x: Vec<Vec<f64>>, y: Vec<f64>, w: Vec<f64>, b: f64, alpha: f64, num_iters: u64) -> PyResult<Lrobj_multi> {
    let mut j_hist : Vec<f64> = vec![];
    let mut mod_w : Vec<f64> = vec![];
    let mut mod_b : f64 = 0.0;
    for i in 0..num_iters {
        let (dj_dw, dj_db) : (Vec<f64>, f64) = compute_gradient_multi(x.clone(), y.clone(), w.clone(), b);
        for (ai, bi) in w.iter().zip(dj_dw.iter()) {
            mod_w.push(ai - alpha * bi);
        }
        mod_b = b - alpha * dj_db;

        if i < 100000 {
            j_hist.push(compute_cost_multi(x.clone(), y.clone(), w.clone(), b));
        }

        if i % (num_iters / 10) == 0 {
            let j_hist_last = j_hist[j_hist.len() - 1];
            println!("Iteration {i}:  Cost: {j_hist_last}");
        }
    }
    Ok(Lrobj_multi::new(mod_w, mod_b, j_hist))
}

#[pyfunction]
fn predict_single_value_poly(x: f64, w: Vec<f64>, b: f64) -> f64 {
    w.iter().zip(0..w.len()).fold(b, |acc, (&wi, ii)| acc + wi * x.powi(ii as i32))
}

#[pyfunction]
fn compute_cost_poly(x: Vec<f64>, y: Vec<f64>, w: Vec<f64>, b: f64) -> f64 {
    let mut cost : f64 = 0.0;
    for i in 0..x.len() {
        cost += (predict_single_value_poly(x[i], w.clone(), b) - y[i]).powi(2);
    }
    cost / (2.0 * x.len() as f64)
}

#[pyfunction]
fn compute_gradient_poly(x: Vec<f64>, y: Vec<f64>, w: Vec<f64>, b: f64) -> (Vec<f64>, f64) {
    let mut dj_dw : Vec<f64> = vec![0.0 ; w.len()];
    let mut dj_db : f64 = 0.0;
    let m = x.len();
    let n = w.len();
    for i in 0..m {
        let err = predict_single_value_poly(x[i], w.clone(), b) - y[i];
        for j in 1..n {
            dj_dw[j] += err * (x[i].powi(j as i32));
            dj_db += err;
        }
    }
    (dj_dw.iter().map(|&a| a / m as f64).collect(), dj_db / m as f64)
}

#[pyfunction]
fn gradient_descent_poly(x: Vec<f64>, y: Vec<f64>, degree: usize, alpha: f64, num_iters: u64) -> PyResult<Lrobj_multi> {
    let mut j_hist : Vec<f64> = vec![];
    let mut w : Vec<f64> = vec![0.0 ; degree+1];
    let mut b = 0.0;
    for i in 0..num_iters {
        let (dj_dw, dj_db) : (Vec<f64>, f64) = compute_gradient_poly(x.clone(), y.clone(), w.clone(), b);
        w = w.iter().zip(dj_dw.iter()).map(|(&v1, &v2)| v1 - alpha * v2).collect();
        b -= alpha * dj_db;

        if i < 100000 {
            j_hist.push(compute_cost_poly(x.clone(), y.clone(), w.clone(), b));
        }
        if i % (num_iters / 10) == 0 {
            let j_hist_last = j_hist[j_hist.len() - 1];
            println!("Iteration {i}: Cost : {j_hist_last}");
        }
    }
    Ok(Lrobj_multi::new(w, b, j_hist))
}

/// Formats the sum of two numbers as string.
// #[pyfunction]
// fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
//     Ok((a + b).to_string())
// }

/// A Python module implemented in Rust.
#[pymodule]
fn vinhal(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_cost, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_descent, m)?)?;
    m.add_function(wrap_pyfunction!(predict_single_value, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cost_multi, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradient_multi, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_descent_multi, m)?)?;
    m.add_function(wrap_pyfunction!(predict_single_value_poly, m)?)?;
    m.add_function(wrap_pyfunction!(compute_cost_poly, m)?)?;
    m.add_function(wrap_pyfunction!(compute_gradient_poly, m)?)?;
    m.add_function(wrap_pyfunction!(gradient_descent_poly, m)?)?;
    Ok(())
}
