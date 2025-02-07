use std::sync::Arc;

use candle_core::{Device, FloatDType, Result, Tensor, Var};

#[derive(Debug, Clone)]
pub struct LinearModel {
    pub weights: Var,
    pub bias: Var,
}

impl LinearModel {
    pub fn new<T: FloatDType>(num_inputs: usize, sigma: f64, device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            weights: Var::randn(
                T::from_f64(0.),
                T::from_f64(sigma),
                (num_inputs, 1),
                &device,
            )?,
            bias: Var::zeros(1, T::DTYPE, &device)?,
        })
    }

    pub fn forward(&self, samples: &Tensor) -> Result<Tensor> {
        assert!(samples.dtype() == self.weights.dtype());
        samples.matmul(&self.weights)?.broadcast_add(&self.bias)
    }

    pub fn vars(&self) -> Vec<Var> {
        vec![self.weights.clone(), self.bias.clone()]
    }
}
