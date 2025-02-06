use std::sync::Arc;

use candle_core::{Device, Error, Result, Tensor, Var};
use candle_datasets::Batcher;
use learn_deep_learning::synthetic_regression_data::SyntheticRegressionDataBuilder;

#[path = "common/common.rs"]
mod common;

#[derive(Debug, Clone)]
pub struct LinearModel {
    lr: f32,
    weights: Var,
    bias: Var,
}

impl LinearModel {
    pub fn new(num_inputs: usize, lr: f32, sigma: f32, device: Arc<Device>) -> Result<Self> {
        Ok(Self {
            lr,
            weights: Var::randn(0_f32, sigma, (num_inputs, 1), &device)?,
            bias: Var::zeros(1, candle_core::DType::F32, &device)?,
        })
    }

    pub fn forward(&self, samples: &Tensor) -> Result<Tensor> {
        samples.matmul(&self.weights)?.broadcast_add(&self.bias)
    }

    pub fn loss(prediction: &Tensor, targets: &Tensor) -> Result<Tensor> {
        ((prediction - targets)?.powf(2.0)? / 2_f64)?.mean_all()
    }

    pub fn step(&mut self, loss: &Tensor) -> Result<()> {
        let grads = loss.backward()?;
        for param in [&self.weights, &self.bias] {
            let grad = grads.get(param).ok_or(Error::debug("failed to get grad"))?;
            param.set(&(param.as_tensor() - (self.lr as f64 * grad)?)?)?;
        }

        Ok(())
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let max_epoch = 5;

    let device = Arc::new(Device::new_cuda(0)?);

    let data =
        SyntheticRegressionDataBuilder::new(Tensor::from_slice(&[2f32, -3.4], (2, 1), &device)?)
            .device(device.clone())
            .bias(4.2)
            .build()?;

    let mut model = LinearModel::new(2, 0.03, 0.01, device)?;

    let mut train_progress = Vec::new();
    let mut val_progress = Vec::new();

    for _ in 0..max_epoch {
        let training_batcher = Batcher::new2(data.iter(true))
            .batch_size(32)
            .return_last_incomplete_batch(false);
        for (i, batch) in training_batcher.enumerate() {
            let (features, target) = batch?;
            assert!(features.dim(0)? == 32);
            assert!(target.dim(0)? == 32);

            let loss = LinearModel::loss(&model.forward(&features)?, &target)?;
            model.step(&loss)?;

            train_progress.push((i, loss.to_scalar::<f32>()?));
        }

        let validate_batcher = Batcher::new2(data.iter(false))
            .batch_size(32)
            .return_last_incomplete_batch(false);
        for (i, batch) in validate_batcher.enumerate() {
            let (features, target) = batch?;
            let loss = LinearModel::loss(&model.forward(&features)?, &target)?;

            val_progress.push((i, loss.to_scalar::<f32>()?));
        }

        println!(
            "error in estimating weights:\n{}",
            (data.weights() - model.weights.as_tensor())?
        );
        println!(
            "error in estimating bias:\n{}",
            (data.bias() as f64 - model.bias.as_tensor())?
        );
    }

    common::plot_progress(&train_progress, &val_progress, false);

    Ok(())
}
