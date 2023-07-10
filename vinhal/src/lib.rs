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

#[pyfunction]
fn predict_single_value(x: Vec<f64>, w: Vec<f64>, b: f64) -> f64 {
    w.iter().zip(x.iter()).fold(b, |acc, (&wi, &xi)| acc + wi * xi)
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
//    m.add_function(wrap_pyfunction!(gradient_descent, m)?)?;
    Ok(())
}
