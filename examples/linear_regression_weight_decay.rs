use std::sync::Arc;

use candle_core::{Device, Result, Tensor, Var};
use candle_datasets::Batcher;
use candle_nn::Optimizer;
use learn_deep_learning::{
    linear_model::LinearModel, synthetic_regression_data::SyntheticRegressionDataBuilder,
};

#[path = "common/common.rs"]
mod common;

#[derive(Debug, Clone)]
pub struct SgdOpt {
    vars: Vec<Var>,
    learning_rate: f64,
    weight_decay: f64,
}

pub struct SgdOptParam {
    pub learning_rate: f64,
    pub weight_decay: f64,
}

impl Optimizer for SgdOpt {
    type Config = SgdOptParam;

    fn new(vars: Vec<Var>, config: Self::Config) -> Result<Self> {
        Ok(Self {
            vars,
            learning_rate: config.learning_rate,
            weight_decay: config.weight_decay,
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
    let max_epoch = 10;

    let device = Arc::new(Device::new_cuda(0)?);

    let num_inputs = 200;
    let data = SyntheticRegressionDataBuilder::new(Tensor::from_slice(
        &vec![0.01f32; num_inputs],
        (num_inputs, 1),
        &device,
    )?)
    .device(device.clone())
    .bias(0.05)
    .noise(0.01)
    .num_train(20)
    .num_validate(100)
    .build::<f32>()?;

    let model = LinearModel::new::<f32>(num_inputs, 0.01, device)?;
    let mut opt = SgdOpt::new(
        model.vars(),
        SgdOptParam {
            learning_rate: 0.01,
            weight_decay: 3.,
        },
    )?;

    let mut train_progress = Vec::new();
    let mut val_progress = Vec::new();

    for _ in 0..max_epoch {
        let training_batcher = Batcher::new2(data.iter(true))
            .batch_size(5)
            .return_last_incomplete_batch(false);
        for (i, batch) in training_batcher.enumerate() {
            let (features, target) = batch?;

            let loss = (((model.forward(&features)? - target)?.powf(2.0)? / 2_f64)?.mean_all()?
                + (opt.weight_decay / 2.) * model.weights.as_tensor().powf(2.0)?.sum_all()?)?;

            opt.backward_step(&loss)?;

            train_progress.push((i, loss.to_scalar::<f32>()?));
        }

        let validate_batcher = Batcher::new2(data.iter(false))
            .batch_size(5)
            .return_last_incomplete_batch(false);
        for (i, batch) in validate_batcher.enumerate() {
            let (features, target) = batch?;
            let loss = (((model.forward(&features)? - target)?.powf(2.0)? / 2_f64)?.mean_all()?
                + (opt.weight_decay / 2.) * model.weights.as_tensor().powf(2.0)?.sum_all()?)?;

            val_progress.push((i, loss.to_scalar::<f32>()?));
        }

        println!(
            "error in estimating weights:\n{}",
            (data.weights() - model.weights.as_tensor())?.reshape((1, num_inputs))?
        );
        println!(
            "error in estimating bias:\n{}",
            (data.bias() - model.bias.as_tensor())?
        );
    }

    println!(
        "L2 norm of weight = {}",
        model.weights.as_tensor().powf(2.0)?.sum_all()?
    );

    common::plot_progress(&train_progress, &val_progress, true);

    Ok(())
}
