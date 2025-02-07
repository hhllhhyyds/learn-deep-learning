use std::sync::Arc;

use candle_core::{Device, Result, Tensor, Var};
use candle_datasets::Batcher;
use candle_nn::optim::Optimizer;

use learn_deep_learning::linear_model::LinearModel;
use learn_deep_learning::synthetic_regression_data::SyntheticRegressionDataBuilder;

#[path = "common/common.rs"]
mod common;

#[derive(Debug, Clone)]
pub struct SgdOpt {
    vars: Vec<Var>,
    learning_rate: f64,
}

impl Optimizer for SgdOpt {
    type Config = f64;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self> {
        Ok(Self {
            vars,
            learning_rate: config,
        })
    }

    fn step(&mut self, grads: &candle_core::backprop::GradStore) -> Result<()> {
        for var in self.vars.iter() {
            if let Some(grad) = grads.get(var) {
                var.set(&var.sub(&(grad * self.learning_rate)?)?)?;
            }
        }
        Ok(())
    }

    fn learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.learning_rate = lr
    }
}

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let max_epoch = 3;

    let device = Arc::new(Device::new_cuda(0)?);

    let data =
        SyntheticRegressionDataBuilder::new(Tensor::from_slice(&[2f64, -3.4], (2, 1), &device)?)
            .device(device.clone())
            .bias(4.2)
            .build::<f64>()?;

    let model = LinearModel::new::<f64>(2, 0.01, device)?;
    let mut opt = SgdOpt::new(model.vars(), 0.03)?;

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

            let loss = ((model.forward(&features)? - target)?.powf(2.0)? / 2.)?.mean_all()?;
            opt.backward_step(&loss)?;

            train_progress.push((i, loss.to_scalar::<f64>()? as f32));
        }

        let validate_batcher = Batcher::new2(data.iter(false))
            .batch_size(32)
            .return_last_incomplete_batch(false);
        for (i, batch) in validate_batcher.enumerate() {
            let (features, target) = batch?;
            let loss = ((model.forward(&features)? - target)?.powf(2.0)? / 2.)?.mean_all()?;

            val_progress.push((i, loss.to_scalar::<f64>()? as f32));
        }

        println!(
            "error in estimating weights:\n{}",
            (data.weights() - model.weights.as_tensor())?
        );
        println!(
            "error in estimating bias:\n{}",
            (data.bias() - model.bias.as_tensor())?
        );
    }

    common::plot_progress(&train_progress, &val_progress, false);

    Ok(())
}
