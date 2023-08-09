use std::f64::consts::E;

#[derive (Debug, Clone)]
pub struct Matrix {
    pub rows: i32,
    pub cols: i32,
    pub size: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    fn index(&self, x: i32, y: i32) -> usize {
        return (y * self.cols + x) as usize;
    }

    fn sum(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.size {
            sum += self.data[i];
        }
        return sum;
    }

    pub fn debug(&self) {
        println!("{}x{}", self.cols, self.rows);
        for y in 0..self.rows {
            print!("[ ");
            for x in 0..self.cols {
                let i = self.index(x, y);
                print!("{} ", self.data[i]);
            }
            print!("]\n");
        }
        print!("\n");
    }

    pub fn sigmoid(&mut self) {
        for i in 0..self.size {
            self.data[i] = 1.0 / (1.0 + E.powf(-self.data[i]));
        }
    }

    pub fn softmax(&mut self) {
        for i in 0..self.size {
            self.data[i] = E.powf(self.data[i]);
        }

        let sum = self.sum();
        for i in 0..self.size {
            self.data[i] /= sum;
        }
    }

    pub fn mean_squared_error(&mut self, target: &Matrix) -> f64 {
        assert!(self.size == target.size);

        let mut error = 0.0;
        for i in 0..self.size {
            error += f64::powi(target.data[i] - self.data[i], 2);
        }

        return error * (1.0 / self.size as f64);
    }

    pub fn transpose(&mut self) -> Matrix {
        let mut result = Matrix{rows: self.cols,
                                cols: self.rows,
                                size: self.size,
                                data: vec![0.0; self.size]};

        for y in 0..result.rows {
            for x in 0..result.cols {
                let i2 = self.index(y, x);
                let i1 = result.index(x, y);
                result.data[i1] = self.data[i2];
            }
        }

        return result;
    }
}

pub fn subtract(a: Matrix, b: Matrix) -> Matrix {
    assert!(a.size == b.size);
    let mut result = Matrix{rows: a.rows, 
                            cols: a.cols, 
                            size: a.size,
                            data: vec![0.0; a.size]};

    for y in 0..result.rows {
        for x in 0..result.cols {
            let i = result.index(x, y);
            result.data[i] = a.data[i] - b.data[i];
        }
    }

    return result;
}

pub fn scale(a: Matrix, b: i32) -> Matrix {
    let mut result = Matrix{rows: a.rows, 
                            cols: a.cols, 
                            size: a.size,
                            data: vec![0.0; a.size]};

    for i in 0..result.size {
        result.data[i] = a.data[i] * b as f64;
    }

    return result;
}

pub fn dot(a: Matrix, b: Matrix) -> Matrix {
    assert!(a.cols == b.rows);
    let mut result = Matrix{size: 0,
                            cols: b.cols,
                            rows: a.rows,
                            data: Vec::new()};
    result.size = (result.cols * result.rows) as usize;
    result.data.resize(result.size, 0.0);

    for r in 0..a.rows {
        for c in 0..b.cols {
            for i in 0..a.cols {
                let i1 = a.index(i, r);
                let i2 = b.index(c, i);
                let i3 = result.index(c, r);
                result.data[i3] += a.data[i1] * b.data[i2];
            }
        }
    }

    return result;
}
