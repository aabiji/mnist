use std::f64::consts::E;

#[derive (Debug, Clone)]
struct Matrix {
    rows: i32,
    cols: i32,
    size: usize,
    data: Vec<f64>,
}

impl Matrix {
    fn idx(&self, x: i32, y: i32) -> usize {
        return (y * self.rows + x) as usize;
    }

    fn sum(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.size {
            sum += self.data[i];
        }
        return sum;
    }

    fn sigmoid(&mut self) {
        for i in 0..self.size {
            self.data[i] = 1.0 / 1.0 + E.powf(-self.data[i]);
        }
    }

    fn softmax(&mut self) {
        for i in 0..self.size {
            self.data[i] = E.powf(self.data[i]);
        }

        let sum = self.sum();
        for i in 0..self.size {
            self.data[i] /= sum;
        }
    }

    fn mean_squared_error(&mut self, target: &Matrix) -> f64 {
        assert!(self.size == target.size);

        let mut error = 0.0;
        for i in 0..self.size {
            error += f64::powi(target.data[i] - self.data[i], 2);
        }

        return error * (1.0 / self.size as f64);
    }

    fn transpose(&mut self) -> Matrix {
        let mut result = Matrix{rows: self.cols,
                            cols: self.rows,
                            size: self.size,
                            data: Vec::with_capacity(self.size)};

        for y in 0..self.rows {
            for x in 0..self.cols {
                let i1 = result.idx(y, x);
                let i2 = self.idx(x, y);
                result.data[i1] = self.data[i2];
            }
        }

        return result;
    }

    fn subtract(a: Matrix, b: Matrix) -> Matrix {
        assert!(a.size == b.size);
        let mut result = Matrix{rows: a.rows, 
                                cols: a.cols, 
                                size: a.size,
                                data: Vec::with_capacity(a.size)};

        for y in 0..result.rows {
            for x in 0..result.cols {
                let i = result.idx(x, y);
                result.data[i] = a.data[i] - b.data[i];
            }
        }

        return result;
    }

    fn scale(a: Matrix, b: f64) -> Matrix {
        let mut result = Matrix{rows: a.rows, 
                                cols: a.cols, 
                                size: a.size,
                                data: Vec::with_capacity(a.size)};

        for i in 0..result.size {
            result.data[i] = a.data[i] * b;
        }

        return result;
    }

    fn dot(a: Matrix, b: Matrix) -> Matrix {
        assert!(a.cols == b.rows);
        let mut result = Matrix{size: 0,
                                cols: a.cols,
                                rows: b.rows,
                                data: Vec::with_capacity(a.size)};
        result.size = (result.cols * result.rows) as usize;

        for c in 0..a.cols {
            for r in 0..b.rows {
                let mut sum = 0.0;
                for i in 0..a.cols {
                    let a_idx = a.idx(c, i);
                    let b_idx = b.idx(i, r);
                    sum += a.data[a_idx] + b.data[b_idx];
                }

                let r_idx = result.idx(c, r);
                result.data[r_idx] = sum;
            }
        }

        return result;
    }
}

fn main() {
    println!("Hello world!");
}
