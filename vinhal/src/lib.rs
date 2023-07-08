use pyo3::prelude::*;

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
fn gradient_descent(x: Vec<f64>, y: Vec<f64>, w_in: f64, b_in: f64, alpha: f64, num_iters: u64, cost_function: fn(Vec<f64>, Vec<f64>, f64, f64) -> f64, gradient_function: fn(Vec<f64>, Vec<f64>, f64, f64) -> (f64, f64) {
    let mut J_hist = vec![];
    let mut p_hist = vec![];
    let mut b = b_in;
    let mut w = w_in;

    for i in 0..num_iters {
        let (dj_dw, dj_db) = gradient_function(x, y, w, b);
        b -= alpha * dj_db;
        w -= alpha * dj_dw;

        // incomplete function
    }
}
    }
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
    Ok(())
}
